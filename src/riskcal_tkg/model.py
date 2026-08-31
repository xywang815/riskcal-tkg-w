from dataclasses import dataclass
import math
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .metrics import filtered_rank


class ContinuousTimeEncoder(nn.Module):
    """Learn a smooth multiplicative time modulation shared across timestamps."""

    def __init__(
        self,
        num_timestamps: int,
        embedding_dim: int,
        time_scale: int | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_timestamps
        scale = max(num_timestamps - 1, 1) if time_scale is None else max(time_scale, 1)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.projection = nn.Linear(5, embedding_dim, bias=False)
        bound = 1.0 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.projection.weight, -bound, bound)

    def forward(self, timestamp_ids: torch.Tensor) -> torch.Tensor:
        values = timestamp_ids.to(dtype=self.projection.weight.dtype) / self.scale
        features = torch.stack(
            (
                torch.ones_like(values),
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )
        return 1.0 + self.projection(features)


class ContinuousComplexTimeEncoder(nn.Module):
    """Map ordered timestamps to a smooth complex-valued modulation."""

    def __init__(
        self,
        num_timestamps: int,
        embedding_dim: int,
        time_scale: int | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_timestamps
        scale = max(num_timestamps - 1, 1) if time_scale is None else max(time_scale, 1)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.projection = nn.Linear(5, 2 * embedding_dim, bias=False)
        bound = 1.0 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.projection.weight, -bound, bound)

    def forward(self, timestamp_ids: torch.Tensor) -> torch.Tensor:
        values = timestamp_ids.to(dtype=self.projection.weight.dtype) / self.scale
        features = torch.stack(
            (
                torch.ones_like(values),
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )
        modulation = self.projection(features)
        real, imaginary = modulation.chunk(2, dim=-1)
        return torch.cat((1.0 + real, imaginary), dim=-1)


class TemporalDistMult(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        embedding_dim: int,
        time_scale: int | None = None,
    ) -> None:
        super().__init__()
        if min(num_entities, num_relations, num_timestamps, embedding_dim) <= 0:
            raise ValueError("embedding table sizes and dimension must be positive")
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.time = ContinuousTimeEncoder(num_timestamps, embedding_dim, time_scale)
        bound = 1.0 / math.sqrt(embedding_dim)
        for embedding in (self.entity, self.relation):
            nn.init.uniform_(embedding.weight, -bound, bound)

    def _validate_quadruples(self, quadruples: torch.Tensor) -> None:
        if quadruples.ndim != 2 or quadruples.shape[1] != 4:
            raise ValueError("quadruples must have shape (n, 4)")
        if quadruples.dtype != torch.long:
            raise ValueError("quadruples must use torch.long IDs")
        if quadruples.numel() == 0:
            return
        s, r, o, t = quadruples.unbind(dim=1)
        if min(int(s.min()), int(o.min())) < 0 or max(int(s.max()), int(o.max())) >= self.entity.num_embeddings:
            raise ValueError("entity ID out of range")
        if int(r.min()) < 0 or int(r.max()) >= self.relation.num_embeddings:
            raise ValueError("relation ID out of range")
        if int(t.min()) < 0:
            raise ValueError("timestamp ID out of range")

    def score_quadruples(self, quadruples: torch.Tensor) -> torch.Tensor:
        self._validate_quadruples(quadruples)
        s, r, o, t = quadruples.unbind(dim=1)
        scores = (
            self.entity(s)
            * self.relation(r)
            * self.entity(o)
            * self.time(t)
        ).sum(dim=1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite quadruple score")
        return scores

    def score_all_objects(self, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries must have shape (n, 3)")
        if queries.dtype != torch.long:
            raise ValueError("queries must use torch.long IDs")
        s, r, t = queries.unbind(dim=1)
        if queries.numel():
            if int(s.min()) < 0 or int(s.max()) >= self.entity.num_embeddings:
                raise ValueError("entity ID out of range")
            if int(r.min()) < 0 or int(r.max()) >= self.relation.num_embeddings:
                raise ValueError("relation ID out of range")
            if int(t.min()) < 0:
                raise ValueError("timestamp ID out of range")
        query_embeddings = self.entity(s) * self.relation(r) * self.time(t)
        scores = query_embeddings @ self.entity.weight.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite all-object score")
        return scores


class ContinuousTemporalComplEx(nn.Module):
    """ComplEx-style temporal scorer with a continuous complex time map."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        embedding_dim: int,
        time_scale: int | None = None,
    ) -> None:
        super().__init__()
        if min(num_entities, num_relations, num_timestamps, embedding_dim) <= 0:
            raise ValueError("embedding table sizes and dimension must be positive")
        self.embedding_dim = embedding_dim
        self.entity = nn.Embedding(num_entities, 2 * embedding_dim)
        self.relation = nn.Embedding(num_relations, 2 * embedding_dim)
        self.time = ContinuousComplexTimeEncoder(
            num_timestamps,
            embedding_dim,
            time_scale,
        )
        bound = 1.0 / math.sqrt(embedding_dim)
        for embedding in (self.entity, self.relation):
            nn.init.uniform_(embedding.weight, -bound, bound)

    def _validate_quadruples(self, quadruples: torch.Tensor) -> None:
        if quadruples.ndim != 2 or quadruples.shape[1] != 4:
            raise ValueError("quadruples must have shape (n, 4)")
        if quadruples.dtype != torch.long:
            raise ValueError("quadruples must use torch.long IDs")
        if quadruples.numel() == 0:
            return
        s, r, o, t = quadruples.unbind(dim=1)
        if min(int(s.min()), int(o.min())) < 0 or max(int(s.max()), int(o.max())) >= self.entity.num_embeddings:
            raise ValueError("entity ID out of range")
        if int(r.min()) < 0 or int(r.max()) >= self.relation.num_embeddings:
            raise ValueError("relation ID out of range")
        if int(t.min()) < 0:
            raise ValueError("timestamp ID out of range")

    @staticmethod
    def _complex_product(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_real, left_imaginary = left.chunk(2, dim=-1)
        right_real, right_imaginary = right.chunk(2, dim=-1)
        return (
            left_real * right_real - left_imaginary * right_imaginary,
            left_real * right_imaginary + left_imaginary * right_real,
        )

    def _query_embedding(
        self,
        subjects: torch.Tensor,
        relations: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_real, relation_imaginary = self._complex_product(
            self.relation(relations),
            self.time(timestamps),
        )
        subject_real, subject_imaginary = self.entity(subjects).chunk(2, dim=-1)
        return (
            subject_real * relation_real - subject_imaginary * relation_imaginary,
            subject_real * relation_imaginary + subject_imaginary * relation_real,
        )

    def score_quadruples(self, quadruples: torch.Tensor) -> torch.Tensor:
        self._validate_quadruples(quadruples)
        subjects, relations, objects, timestamps = quadruples.unbind(dim=1)
        query_real, query_imaginary = self._query_embedding(
            subjects,
            relations,
            timestamps,
        )
        object_real, object_imaginary = self.entity(objects).chunk(2, dim=-1)
        scores = (
            query_real * object_real + query_imaginary * object_imaginary
        ).sum(dim=1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite quadruple score")
        return scores

    def score_all_objects(self, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries must have shape (n, 3)")
        if queries.dtype != torch.long:
            raise ValueError("queries must use torch.long IDs")
        subjects, relations, timestamps = queries.unbind(dim=1)
        if queries.numel():
            if int(subjects.min()) < 0 or int(subjects.max()) >= self.entity.num_embeddings:
                raise ValueError("entity ID out of range")
            if int(relations.min()) < 0 or int(relations.max()) >= self.relation.num_embeddings:
                raise ValueError("relation ID out of range")
            if int(timestamps.min()) < 0:
                raise ValueError("timestamp ID out of range")
        query_real, query_imaginary = self._query_embedding(
            subjects,
            relations,
            timestamps,
        )
        entity_real, entity_imaginary = self.entity.weight.chunk(2, dim=-1)
        scores = (
            query_real @ entity_real.T
            + query_imaginary @ entity_imaginary.T
        )
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite all-object score")
        return scores


TemporalModel = TemporalDistMult | ContinuousTemporalComplEx


def build_temporal_model(
    model_name: str,
    num_entities: int,
    num_relations: int,
    num_timestamps: int,
    embedding_dim: int,
    time_scale: int | None = None,
) -> TemporalModel:
    models = {
        "temporal_distmult": TemporalDistMult,
        "continuous_tcomplex": ContinuousTemporalComplEx,
    }
    try:
        model_class = models[model_name]
    except KeyError as error:
        raise ValueError(f"unknown model_name: {model_name}") from error
    return model_class(
        num_entities,
        num_relations,
        num_timestamps,
        embedding_dim,
        time_scale,
    )


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "temporal_distmult"
    negative_sampling: str = "uniform"
    embedding_dim: int = 128
    epochs: int = 100
    batch_size: int = 512
    negatives: int = 64
    margin: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    seed: int = 17
    eval_every: int = 5
    patience: int = 10

    def __post_init__(self) -> None:
        if self.model_name not in {"temporal_distmult", "continuous_tcomplex"}:
            raise ValueError("unknown model_name")
        if self.negative_sampling not in {"uniform", "filtered"}:
            raise ValueError("negative_sampling must be uniform or filtered")
        if min(
            self.embedding_dim,
            self.epochs,
            self.batch_size,
            self.negatives,
            self.eval_every,
            self.patience,
        ) <= 0:
            raise ValueError("training sizes must be positive")
        if self.margin < 0:
            raise ValueError("training margin must be nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")


@dataclass(frozen=True)
class TrainingResult:
    model: TemporalModel
    loss_history: tuple[float, ...]
    epochs_trained: int
    best_epoch: int
    best_validation_mrr: float | None
    epoch_seconds: tuple[float, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(device)


def _validation_mrr(
    model: TemporalModel,
    facts: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    truth: dict[tuple[int, int, int], set[int]] = {}
    for subject, relation, object_, timestamp in facts:
        truth.setdefault(
            (int(subject), int(relation), int(timestamp)), set()
        ).add(int(object_))
    ranks: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(facts), batch_size):
            batch = facts[start : start + batch_size]
            queries = torch.as_tensor(
                batch[:, [0, 1, 3]], dtype=torch.long, device=device
            )
            scores = model.score_all_objects(queries).cpu().numpy()
            for index, (subject, relation, object_, timestamp) in enumerate(batch):
                other_true = truth[
                    (int(subject), int(relation), int(timestamp))
                ] - {int(object_)}
                ranks.append(
                    filtered_rank(scores[index], int(object_), other_true)
                )
    model.train()
    return float(np.mean(1.0 / np.asarray(ranks, dtype=float)))


def _encoded_fact_keys(
    facts: torch.Tensor,
    num_entities: int,
    num_relations: int,
    num_timestamps: int,
) -> torch.Tensor:
    subjects, relations, objects, timestamps = facts.unbind(dim=1)
    query_keys = (subjects * num_relations + relations) * num_timestamps + timestamps
    return query_keys * num_entities + objects


def _sample_negative_objects(
    positive: torch.Tensor,
    *,
    num_entities: int,
    num_relations: int,
    num_timestamps: int,
    negatives: int,
    generator: torch.Generator,
    negative_sampling: str,
    known_positive_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    if negative_sampling not in {"uniform", "filtered"}:
        raise ValueError("negative_sampling must be uniform or filtered")
    if positive.device.type != "cpu":
        raise ValueError("negative sampling expects CPU facts")
    repeated = positive.repeat_interleave(negatives, dim=0)
    true_objects = repeated[:, 2]
    sampled = torch.randint(
        num_entities,
        (len(repeated),),
        generator=generator,
        dtype=torch.long,
    )
    for _ in range(100):
        if negative_sampling == "uniform":
            rejected = sampled == true_objects if num_entities > 1 else torch.zeros_like(sampled, dtype=torch.bool)
        else:
            if known_positive_keys is None or len(known_positive_keys) == 0:
                raise ValueError("filtered sampling requires known positive keys")
            candidates = repeated.clone()
            candidates[:, 2] = sampled
            keys = _encoded_fact_keys(
                candidates,
                num_entities,
                num_relations,
                num_timestamps,
            )
            positions = torch.searchsorted(known_positive_keys, keys)
            valid_positions = positions < len(known_positive_keys)
            rejected = torch.zeros_like(valid_positions)
            rejected[valid_positions] = (
                known_positive_keys[positions[valid_positions]]
                == keys[valid_positions]
            )
        if not bool(rejected.any()):
            return sampled
        sampled[rejected] = torch.randint(
            num_entities,
            (int(rejected.sum()),),
            generator=generator,
            dtype=torch.long,
        )
    raise RuntimeError("could not draw a valid negative after 100 attempts")


def train_model(
    facts: np.ndarray,
    num_entities: int,
    num_relations: int,
    num_timestamps: int,
    config: TrainingConfig,
    device: str = "auto",
    validation_facts: np.ndarray | None = None,
) -> TrainingResult:
    if facts.ndim != 2 or facts.shape[1] != 4 or len(facts) == 0:
        raise ValueError("facts must be a nonempty array with shape (n, 4)")
    seed_everything(config.seed)
    target_device = _resolve_device(device)
    model = build_temporal_model(
        config.model_name,
        num_entities,
        num_relations,
        num_timestamps,
        config.embedding_dim,
        time_scale=int(facts[:, 3].max()),
    ).to(target_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    fact_tensor = torch.as_tensor(facts, dtype=torch.long)
    known_positive_keys = None
    if config.negative_sampling == "filtered":
        known_positive_keys = torch.unique(
            _encoded_fact_keys(
                fact_tensor,
                num_entities,
                num_relations,
                num_timestamps,
            ),
            sorted=True,
        )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    loss_history: list[float] = []
    epoch_seconds: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_mrr = -math.inf
    evaluations_without_improvement = 0

    for epoch_index in range(config.epochs):
        epoch_started = time.perf_counter()
        permutation = torch.randperm(len(fact_tensor), generator=generator)
        total_loss = 0.0
        observed = 0
        for start in range(0, len(fact_tensor), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            positive_cpu = fact_tensor[indices]
            sampled_objects = _sample_negative_objects(
                positive_cpu,
                num_entities=num_entities,
                num_relations=num_relations,
                num_timestamps=num_timestamps,
                negatives=config.negatives,
                generator=generator,
                negative_sampling=config.negative_sampling,
                known_positive_keys=known_positive_keys,
            )
            positive = positive_cpu.to(target_device)
            negative = positive.repeat_interleave(config.negatives, dim=0)
            negative[:, 2] = sampled_objects.to(target_device)

            positive_scores = model.score_quadruples(positive)
            negative_scores = model.score_quadruples(negative).view(
                len(positive),
                config.negatives,
            )
            loss = F.softplus(
                config.margin + negative_scores - positive_scores.unsqueeze(1)
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * len(positive)
            observed += len(positive)
        loss_history.append(total_loss / observed)
        epoch_seconds.append(time.perf_counter() - epoch_started)

        if validation_facts is not None and (
            (epoch_index + 1) % config.eval_every == 0
            or epoch_index + 1 == config.epochs
        ):
            validation_mrr = _validation_mrr(
                model,
                validation_facts,
                config.batch_size,
                target_device,
            )
            if validation_mrr > best_validation_mrr + 1e-12:
                best_validation_mrr = validation_mrr
                best_epoch = epoch_index + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                if evaluations_without_improvement >= config.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_epoch == 0:
        best_epoch = len(loss_history)
    model = model.to("cpu")
    return TrainingResult(
        model=model,
        loss_history=tuple(loss_history),
        epochs_trained=len(loss_history),
        best_epoch=best_epoch,
        best_validation_mrr=(
            None if best_validation_mrr == -math.inf else best_validation_mrr
        ),
        epoch_seconds=tuple(epoch_seconds),
    )
