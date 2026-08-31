import pytest
import torch
import numpy as np

from riskcal_tkg.model import (
    ContinuousTemporalComplEx,
    TemporalDistMult,
    TrainingConfig,
    _encoded_fact_keys,
    _sample_negative_objects,
    train_model,
)


def test_temporal_distmult_matches_hand_calculation() -> None:
    model = TemporalDistMult(3, 2, 4, embedding_dim=2)
    with torch.no_grad():
        model.entity.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        model.relation.weight.copy_(torch.tensor([[2.0, 1.0], [1.0, 3.0]]))
        model.time.projection.weight.zero_()
        model.time.projection.weight[:, 0].copy_(torch.tensor([1.0, 2.0]))
    score = model.score_quadruples(torch.tensor([[0, 0, 1, 0]]))
    assert torch.allclose(score, torch.tensor([36.0]))


def test_score_all_objects_matches_individual_scores() -> None:
    torch.manual_seed(17)
    model = TemporalDistMult(4, 2, 3, embedding_dim=5)
    query = torch.tensor([[0, 1, 2]])
    all_scores = model.score_all_objects(query)
    rows = torch.tensor([[0, 1, object_id, 2] for object_id in range(4)])
    assert torch.allclose(all_scores[0], model.score_quadruples(rows))


def test_continuous_tcomplex_matches_hand_calculation() -> None:
    model = ContinuousTemporalComplEx(2, 1, 2, embedding_dim=1)
    with torch.no_grad():
        model.entity.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.relation.weight.copy_(torch.tensor([[5.0, 6.0]]))
        model.time.projection.weight.zero_()
    score = model.score_quadruples(torch.tensor([[0, 0, 1, 0]]))
    assert torch.allclose(score, torch.tensor([43.0]))
    all_scores = model.score_all_objects(torch.tensor([[0, 0, 0]]))
    rows = torch.tensor([[0, 0, object_id, 0] for object_id in range(2)])
    assert torch.allclose(all_scores[0], model.score_quadruples(rows))


def test_time_encoder_parameter_count_does_not_grow_with_future_timestamps() -> None:
    short_horizon = TemporalDistMult(4, 2, 4, embedding_dim=8)
    long_horizon = TemporalDistMult(4, 2, 400, embedding_dim=8)
    short_count = sum(parameter.numel() for parameter in short_horizon.time.parameters())
    long_count = sum(parameter.numel() for parameter in long_horizon.time.parameters())
    assert short_count == long_count


def test_time_encoder_uses_training_scale_and_accepts_unknown_future_ids() -> None:
    facts = np.asarray([[0, 0, 1, 0], [1, 0, 0, 2]], dtype=np.int64)
    result = train_model(
        facts,
        num_entities=2,
        num_relations=1,
        num_timestamps=100,
        config=TrainingConfig(
            embedding_dim=4,
            epochs=1,
            batch_size=2,
            negatives=1,
            seed=17,
        ),
        device="cpu",
    )
    assert result.model.time.scale.item() == 2.0
    score = result.model.score_quadruples(torch.tensor([[0, 0, 1, 1000]]))
    assert torch.isfinite(score).all()


def test_model_rejects_out_of_range_ids() -> None:
    model = TemporalDistMult(3, 2, 4, embedding_dim=2)
    with pytest.raises(ValueError, match="entity"):
        model.score_quadruples(torch.tensor([[0, 0, 3, 0]]))


def test_training_reduces_loss_on_toy_facts() -> None:
    facts = np.asarray(
        [
            [0, 0, 1, 0],
            [1, 0, 2, 1],
            [2, 0, 0, 2],
            [0, 1, 2, 1],
            [2, 1, 1, 2],
        ],
        dtype=np.int64,
    )
    result = train_model(
        facts,
        num_entities=3,
        num_relations=2,
        num_timestamps=3,
        config=TrainingConfig(
            embedding_dim=8,
            epochs=20,
            batch_size=5,
            negatives=4,
            learning_rate=0.05,
            seed=17,
        ),
        device="cpu",
    )
    assert result.loss_history[-1] < result.loss_history[0]
    assert all(np.isfinite(result.loss_history))


def test_training_keeps_all_object_scores_distinguishable() -> None:
    facts = np.asarray(
        [
            [0, 0, 1, 0],
            [1, 0, 2, 1],
            [2, 0, 0, 2],
            [0, 1, 2, 1],
            [2, 1, 1, 2],
        ],
        dtype=np.int64,
    )
    result = train_model(
        facts,
        num_entities=3,
        num_relations=2,
        num_timestamps=3,
        config=TrainingConfig(
            embedding_dim=8,
            epochs=20,
            batch_size=5,
            negatives=4,
            learning_rate=0.05,
            seed=17,
        ),
        device="cpu",
    )
    scores = result.model.score_all_objects(torch.tensor([[0, 0, 0], [1, 0, 1]]))
    assert torch.std(scores, dim=1).min() > 0.0


def test_training_is_reproducible_for_fixed_seed() -> None:
    facts = np.asarray([[0, 0, 1, 0], [1, 0, 0, 1]], dtype=np.int64)
    config = TrainingConfig(
        embedding_dim=4,
        epochs=2,
        batch_size=2,
        negatives=2,
        learning_rate=0.01,
        seed=29,
    )
    first = train_model(facts, 2, 1, 2, config, device="cpu")
    second = train_model(facts, 2, 1, 2, config, device="cpu")
    assert first.loss_history == second.loss_history
    assert torch.equal(first.model.entity.weight, second.model.entity.weight)


def test_filtered_negative_sampling_rejects_all_training_positives() -> None:
    facts = torch.tensor([[0, 0, 1, 0], [0, 0, 2, 0]], dtype=torch.long)
    keys = torch.unique(_encoded_fact_keys(facts, 4, 1, 1), sorted=True)
    sampled = _sample_negative_objects(
        facts[:1],
        num_entities=4,
        num_relations=1,
        num_timestamps=1,
        negatives=64,
        generator=torch.Generator().manual_seed(17),
        negative_sampling="filtered",
        known_positive_keys=keys,
    )
    assert set(sampled.tolist()) <= {0, 3}


def test_training_supports_tcomplex_and_filtered_negatives() -> None:
    facts = np.asarray(
        [[0, 0, 1, 0], [0, 0, 2, 0], [1, 0, 0, 1]],
        dtype=np.int64,
    )
    result = train_model(
        facts,
        num_entities=3,
        num_relations=1,
        num_timestamps=2,
        config=TrainingConfig(
            model_name="continuous_tcomplex",
            negative_sampling="filtered",
            embedding_dim=4,
            epochs=2,
            batch_size=3,
            negatives=2,
            seed=17,
        ),
        device="cpu",
    )
    assert isinstance(result.model, ContinuousTemporalComplEx)
    assert all(np.isfinite(result.loss_history))


def test_validation_mrr_enables_early_stopping() -> None:
    facts = np.asarray(
        [[0, 0, 1, 0], [1, 0, 2, 1], [2, 0, 0, 2]], dtype=np.int64
    )
    result = train_model(
        facts,
        num_entities=3,
        num_relations=1,
        num_timestamps=3,
        config=TrainingConfig(
            embedding_dim=4,
            epochs=50,
            batch_size=3,
            negatives=2,
            learning_rate=1e-12,
            eval_every=1,
            patience=1,
            seed=17,
        ),
        device="cpu",
        validation_facts=facts,
    )
    assert result.epochs_trained < 50
    assert 1 <= result.best_epoch <= result.epochs_trained
