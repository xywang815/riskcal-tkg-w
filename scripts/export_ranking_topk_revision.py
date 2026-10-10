"""Restore exact test ranks and evaluate pure Top-K on verified frozen scorers.

Use explicit matrix, gate and utility roots, or --base for the legacy layout.
The per-label NPZ files plus dataset fact indexes permit local budget reanalysis.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from riskcal_tkg.artifacts import load_verified_checkpoint, sha256_file
from riskcal_tkg.config import load_config
from riskcal_tkg.experiment import _score_all_objects, _config_hash_payload, _jsonable
from riskcal_tkg.followup import build_query_grouping
from riskcal_tkg.metrics import filtered_rank
from riskcal_tkg.model import build_temporal_model

FACT = ['subject_id', 'relation_id', 'true_object_id', 'timestamp']
SORT = ['timestamp', 'subject_id', 'relation_id', 'true_object_id']
QUERY = ['subject_id', 'relation_id', 'timestamp']
SEEDS = [17, 29, 43, 59, 71]
BUDGETS = [50, 100, 250, 500, 1000]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def restore_raw_ranks(frame):
    ordered = frame.sort_values(QUERY + ['true_score', 'true_object_id'],
                                ascending=[True, True, True, False, True])
    predecessors = ordered.groupby(QUERY, sort=False).cumcount().reindex(frame.index)
    return frame['rank'].to_numpy(np.int64) + predecessors.to_numpy(np.int64)


def synthetic_check():
    # Exhaust all nonempty answer subsets for tied and non-tied score vectors.
    count = 0
    for scores in (np.array([2., 2., 1., 0.]), np.array([1., 3., 0., 2.]), np.zeros(4)):
        actual = np.argsort(np.argsort(-scores, kind='stable'), kind='stable') + 1
        for bits in range(1, 16):
            truth = {i for i in range(4) if bits & (1 << i)}
            rows = [dict(subject_id=0, relation_id=0, timestamp=0, true_object_id=i,
                         true_score=scores[i], rank=filtered_rank(scores, i, truth - {i})) for i in truth]
            frame = pd.DataFrame(rows)
            require(np.array_equal(restore_raw_ranks(frame), actual[frame.true_object_id]), 'synthetic rank failure')
            count += len(rows)
    return count


def coverage_counts(covered, grouping):
    counts = np.bincount(grouping.inverse, weights=covered, minlength=grouping.query_count)
    full = counts == grouping.answer_counts
    single = grouping.answer_counts == 1
    return dict(label_count=len(covered), covered_label_count=int(np.sum(covered)),
                query_count=grouping.query_count, full_query_count=int(full.sum()),
                single_query_count=int(single.sum()), single_full_query_count=int(full[single].sum()),
                multi_query_count=int((~single).sum()), multi_full_query_count=int(full[~single].sum()),
                partial_recall_sum=float(np.sum(counts / grouping.answer_counts)))


def evaluate(job):
    case, seed, record, gate_str, utility_str, out_str, specification_hash = job
    start = time.monotonic()
    torch.set_num_threads(2)
    gate, utility, out = Path(gate_str), Path(utility_str), Path(out_str)
    key = f'{case}_seed{seed}'
    marker_out = out / 'conditions' / f'{key}.complete.json'
    if marker_out.exists():
        previous = json.loads(marker_out.read_text())
        require(previous['specification_sha256'] == specification_hash, 'changed resume specification')
        for name, digest in previous['output_sha256'].items():
            require(sha256_file(out / name) == digest, 'changed completed output')
        return previous
    root = Path(record['run_root'])
    cfg_path = Path(record['config'])
    require(sha256_file(cfg_path) == record['config_sha256'], 'configuration file hash')
    cfg = load_config(cfg_path)
    config_hash = hashlib.sha256(json.dumps(_config_hash_payload(_jsonable(asdict(cfg))), sort_keys=True).encode()).hexdigest()
    marker_path = root / 'conditions' / f'seed{seed}_delete0p00.complete.json'
    marker = json.loads(marker_path.read_text())
    dataset = json.loads((root / 'dataset_manifest.json').read_text())
    checkpoint = root / 'checkpoints' / f'seed{seed}_delete0p00.ckpt'
    queries = root / 'conditions' / f'seed{seed}_delete0p00.queries.parquet'
    windows = root / 'conditions' / f'seed{seed}_delete0p00.windows.csv'
    sources = {}
    for field, path in [('checkpoint', checkpoint), ('queries', queries), ('windows', windows)]:
        digest = sha256_file(path)
        require(digest == marker['artifacts'][field], f'{field} source hash')
        sources[str(path)] = digest
    require(marker['status'] == 'complete' and marker['config_sha256'] == config_hash and marker['dataset_sha256'] == dataset['sha256'], 'training identity')
    gm = json.loads((gate / 'conditions' / f'{key}.complete.json').read_text())
    um = json.loads((utility / 'conditions' / f'{key}.complete.json').read_text())
    cache_path = gate / 'scores' / f'{key}.parquet'
    gate_csv = gate / 'conditions' / f'{key}.csv'
    utility_path = utility / 'conditions' / f'{key}.parquet'
    for path, expected in [(cache_path, gm['scores_sha256']), (gate_csv, gm['output_sha256']), (utility_path, um['outputs']['timestamps'])]:
        digest = sha256_file(path)
        require(digest == expected, 'cached source hash')
        sources[str(path)] = digest
    for m in [gm, um]:
        require(m['status'] == 'complete' and m['checkpoint_sha256'] == marker['artifacts']['checkpoint'] and m['config_sha256'] == config_hash and m['dataset_sha256'] == dataset['sha256'], 'stage identity')
    q = pd.read_parquet(queries, filters=[('method', '==', 'top1')],
                        columns=FACT + ['rank', 'answer_count', 'true_score', 'top1_correct'])
    q = q.sort_values(SORT).reset_index(drop=True)
    require(not q.duplicated(FACT).any(), 'duplicate test facts')
    observed_counts = q.groupby(QUERY)['true_object_id'].transform('size')
    require(np.array_equal(observed_counts, q.answer_count), 'incomplete true-answer group')
    raw_rank = restore_raw_ranks(q)
    vocab, relations = int(dataset['num_entities']), int(dataset['num_relations'])
    require(np.all((raw_rank >= q['rank'].to_numpy()) & (raw_rank <= vocab)), 'rank range')
    require(np.array_equal(raw_rank == 1, q.top1_correct.to_numpy()), 'Top-1 replay mismatch')
    cached = pd.read_parquet(cache_path)
    cached = cached[cached.phase == 'test'].sort_values(SORT).reset_index(drop=True)
    require(np.array_equal(q[FACT].to_numpy(), cached[FACT].to_numpy()), 'test cache alignment')
    u = pd.read_parquet(utility_path)
    u = u[u.method == 'label_minmax_rolling'].set_index('timestamp').sort_index()
    require(set(q.timestamp.unique()) == set(u.index), 'utility timestamp coverage')
    eligible = cached['minmax'].to_numpy() <= q.timestamp.map(u.threshold).to_numpy()
    original_windows = pd.read_csv(windows)
    original_windows = original_windows[original_windows.method == 'top1'].set_index('timestamp')
    facts = q[FACT].to_numpy(dtype=np.int64)
    payload = load_verified_checkpoint(checkpoint, config_sha256=config_hash,
                                      dataset_sha256=dataset['sha256'], deletion_mask_sha256=marker['deletion_mask_sha256'])
    model = build_temporal_model(cfg.model_name, vocab, 2 * relations, int(dataset['num_timestamps']),
                                 cfg.embedding_dim, time_scale=int(float(payload['time_scale'])), time_encoding=cfg.time_encoding)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to('cpu').eval()
    dates = sorted(q.timestamp.unique())
    replay_dates = [dates[0], dates[len(dates) // 2], dates[-1]]
    replay_facts = replay_queries = 0
    max_true_score_difference = max_window_mrr_difference = 0.
    timestamp_rows = []
    for timestamp, frame in q.groupby('timestamp', sort=True):
        indices = frame.index.to_numpy()
        f = facts[indices]
        group = build_query_grouping(f)
        ranks = raw_rank[indices]
        window_diff = abs(float(np.mean(1. / frame['rank'].to_numpy())) - float(original_windows.loc[timestamp, 'mrr']))
        max_window_mrr_difference = max(max_window_mrr_difference, window_diff)
        require(window_diff < 1e-12 and len(frame) == int(original_windows.loc[timestamp, 'query_count']), 'original MRR/count mismatch')
        if timestamp in replay_dates:
            # Preserve original/cache fact order and batch shapes during score replay.
            original_cache = pd.read_parquet(cache_path, filters=[('phase', '==', 'test'), ('timestamp', '==', int(timestamp))])
            original_facts = original_cache[FACT].to_numpy(np.int64)
            original_scores = _score_all_objects(model, original_facts, cfg.batch_size)
            lookup = pd.DataFrame(original_facts, columns=FACT).reset_index().set_index(FACT)['index']
            reorder = lookup.reindex(pd.MultiIndex.from_frame(frame[FACT])).to_numpy(np.int64)
            score_matrix = original_scores[reorder]
            true_scores = score_matrix[np.arange(len(f)), f[:, 2]]
            score_diff = float(np.max(np.abs(true_scores.astype(float) - frame.true_score.to_numpy())))
            max_true_score_difference = max(max_true_score_difference, score_diff)
            require(score_diff == 0., 'true score replay changed')
            # Direct raw ranks by pairwise greater score / entity-ID tie comparisons.
            ids = np.arange(vocab)
            direct_raw = 1 + np.sum(score_matrix > true_scores[:, None], axis=1) + np.sum((score_matrix == true_scores[:, None]) & (ids[None, :] < f[:, 2, None]), axis=1)
            require(np.array_equal(direct_raw, ranks), 'direct raw rank mismatch')
            # Independent filtered replay through the original metric implementation.
            truth = [set(f[group.inverse == j, 2].tolist()) for j in range(group.query_count)]
            direct_filtered = [filtered_rank(score_matrix[i], int(f[i, 2]), truth[group.inverse[i]] - {int(f[i, 2])}) for i in range(len(f))]
            require(np.array_equal(direct_filtered, frame['rank'].to_numpy()), 'direct filtered rank mismatch')
            replay_facts += len(f)
            replay_queries += group.query_count
        for budget in BUDGETS:
            top = coverage_counts(ranks <= budget, group)
            capped = coverage_counts((ranks <= budget) & eligible[indices], group)
            pref = f'budget_{budget}_'
            for name, value in capped.items():
                require(abs(value - float(u.loc[timestamp, pref + name])) < 1e-9, f'Minmax count mismatch {timestamp}/{budget}/{name}')
            row = dict(case=case, seed=seed, timestamp=int(timestamp), budget=budget, vocabulary_size=vocab,
                       reciprocal_rank_sum=float(np.sum(1. / frame['rank'].to_numpy())), raw_reciprocal_rank_sum=float(np.sum(1. / ranks)))
            row.update({f'topk_{k}': v for k, v in top.items()})
            row.update({f'minmax_{k}': v for k, v in capped.items()})
            row['topk_set_size_sum'] = budget * group.query_count
            row['minmax_set_size_sum'] = float(u.loc[timestamp, pref + 'set_size_sum'])
            timestamp_rows.append(row)
    timestamps = pd.DataFrame(timestamp_rows)
    metrics = []
    for direction, mask in [('both', np.ones(len(q), bool)), ('tail', q.relation_id.to_numpy() < relations), ('head', q.relation_id.to_numpy() >= relations)]:
        r = q['rank'].to_numpy()[mask]
        metrics.append(dict(case=case, seed=seed, direction=direction, label_count=len(r), filtered_mrr=float(np.mean(1. / r)), raw_mrr=float(np.mean(1. / raw_rank[mask])), **{f'filtered_hits_at_{k}': float(np.mean(r <= k)) for k in [1, 3, 10]}))
    budget_rows = []
    for budget, rows in timestamps.groupby('budget'):
        sums = rows.select_dtypes('number').sum()
        row = dict(case=case, seed=seed, budget=int(budget), vocabulary_size=vocab)
        for method in ['topk', 'minmax']:
            p = method + '_'
            for metric, num, den in [('label_coverage', 'covered_label_count', 'label_count'), ('full_set_coverage', 'full_query_count', 'query_count'), ('multi_full_set_coverage', 'multi_full_query_count', 'multi_query_count'), ('single_full_set_coverage', 'single_full_query_count', 'single_query_count'), ('partial_answer_recall', 'partial_recall_sum', 'query_count'), ('mean_set_size', 'set_size_sum', 'query_count')]:
                row[p + metric] = float(sums[p + num] / sums[p + den])
        row['ranking_exclusion'] = 1 - row['topk_label_coverage']
        row['threshold_exclusion'] = row['topk_label_coverage'] - row['minmax_label_coverage']
        row['candidate_saving_fraction'] = 1 - row['minmax_mean_set_size'] / budget
        require(row['threshold_exclusion'] >= -1e-12, 'set-inclusion contradiction')
        budget_rows.append(row)
    # Store aligned integer ranks, not candidate score matrices or checkpoints.
    dataset_name = 'icews05_15' if case.startswith('icews05_15') else 'icews14'
    fact_bytes = np.asarray(facts, dtype='<i4').tobytes()
    fact_hash = hashlib.sha256(fact_bytes).hexdigest()
    npz_path = out / 'ranks' / f'{key}.npz'
    np.savez_compressed(npz_path, filtered_rank=q['rank'].to_numpy(np.uint16 if vocab < 65536 else np.uint32),
                        raw_rank=raw_rank.astype(np.uint16 if vocab < 65536 else np.uint32), minmax_rolling_eligible=eligible)
    # Identical facts across conditions; use content-addressed file name (atomic write).
    fact_path = out / 'ranks' / f'{dataset_name}_facts_{fact_hash[:12]}.npz'
    tmp = fact_path.with_name(f'.{key}.facts.npz')
    np.savez_compressed(tmp, facts=facts.astype(np.int32), num_relations=np.array(relations))
    tmp.replace(fact_path)
    paths = [npz_path, fact_path]
    for suffix, data in [('timestamps.csv.gz', timestamps), ('ranking.csv', pd.DataFrame(metrics)), ('budgets.csv', pd.DataFrame(budget_rows))]:
        path = out / 'conditions' / f'{key}.{suffix}'
        data.to_csv(path, index=False)
        paths.append(path)
    provenance = dict(case=case, seed=seed, status='complete', specification_sha256=specification_hash,
                      config_sha256=config_hash, dataset_sha256=dataset['sha256'], sources_sha256=sources,
                      fact_columns=FACT, fact_index_file=str(fact_path.relative_to(out)), fact_array_sha256=fact_hash,
                      output_sha256={str(p.relative_to(out)): sha256_file(p) for p in paths},
                      label_count=len(q), query_count=int(q[QUERY].drop_duplicates().shape[0]),
                      replay_dates=list(map(int, replay_dates)), replay_fact_count=replay_facts, replay_query_count=replay_queries,
                      max_true_score_difference=max_true_score_difference, max_original_window_mrr_difference=max_window_mrr_difference,
                      all_budget_timestamp_counts_verified=True, seconds=time.monotonic() - start)
    save_json(marker_out, provenance)
    return provenance


def parse_args(argv=None):
    ap = ArgumentParser()
    ap.add_argument('--base', type=Path, help='Legacy result root; supplies defaults for omitted input roots.')
    ap.add_argument('--matrix-root', type=Path, help='Directory containing matrix_progress.json.')
    ap.add_argument('--gate-root', type=Path, help='Completed normalized-score gate directory.')
    ap.add_argument('--utility-root', type=Path, help='Completed candidate-utility directory.')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--case')
    ap.add_argument('--seed', type=int)
    args = ap.parse_args(argv)
    for field, legacy_relative in (
        ('matrix_root', 'time_encoder_audit_matrix'),
        ('gate_root', 'analysis/normalized_gate_20260908'),
        ('utility_root', 'analysis/utility_0690baf'),
    ):
        if getattr(args, field) is None:
            if args.base is None:
                ap.error(f'--{field.replace("_", "-")} is required when --base is omitted')
            setattr(args, field, args.base / legacy_relative)
    return args


def main():
    args = parse_args()
    out = args.output
    for directory in [out, out / 'conditions', out / 'ranks']:
        directory.mkdir(parents=True, exist_ok=True)
    runs_path = args.matrix_root / 'matrix_progress.json'
    runs = json.loads(runs_path.read_text())['runs']
    require(len(runs) == 16 and all(r['status'] == 'complete' for r in runs.values()), 'incomplete source matrix')
    specification = dict(version=1, script_sha256=sha256_file(Path(__file__)), matrix_sha256=sha256_file(runs_path),
                         cases=sorted(runs), seeds=SEEDS, budgets=BUDGETS, filter='same subject/relation/timestamp recorded true entities',
                         tie_rule='descending score then ascending entity ID', replay='first, middle, last test timestamps in every condition',
                         aggregation='micro over test facts within seed; equal mean and sample SD over five seeds',
                         synthetic_verified_labels=synthetic_check())
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    save_json(out / 'specification.json', specification)
    jobs = [(case, seed, record, str(args.gate_root), str(args.utility_root), str(out), spec_hash) for case, record in sorted(runs.items()) for seed in SEEDS if (not args.case or case == args.case) and (not args.seed or seed == args.seed)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result['case']} seed {result['seed']} verified {result['replay_fact_count']} replay facts; {result['seconds']:.1f}s", flush=True)
            save_json(out / 'progress.json', dict(completed=len(results), expected=len(jobs), latest=result['case'], seed=result['seed']))
    ranking = pd.concat([pd.read_csv(out / 'conditions' / f"{r['case']}_seed{r['seed']}.ranking.csv") for r in results], ignore_index=True)
    budgets = pd.concat([pd.read_csv(out / 'conditions' / f"{r['case']}_seed{r['seed']}.budgets.csv") for r in results], ignore_index=True)
    ranking.sort_values(['case', 'seed', 'direction']).to_csv(out / 'ranking_by_seed.csv', index=False)
    budgets.sort_values(['case', 'seed', 'budget']).to_csv(out / 'topk_by_seed.csv', index=False)
    for name, frame, keys, drop in [('ranking_summary', ranking, ['case', 'direction'], ['seed', 'label_count']), ('topk_summary', budgets, ['case', 'budget'], ['seed', 'vocabulary_size'])]:
        metrics = [c for c in frame.select_dtypes('number').columns if c not in keys + drop]
        summary = frame.groupby(keys)[metrics].agg(['mean', 'std'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(out / f'{name}.csv', index=False)
    save_json(out / 'verification.json', dict(status='complete', condition_count=len(results), full_matrix=len(results) == 80,
        completed_at=datetime.now(UTC).isoformat(), specification_sha256=spec_hash,
        source_runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__),
        replay_fact_count=sum(r['replay_fact_count'] for r in results), replay_timestamp_count=3 * len(results),
        max_true_score_difference=max(r['max_true_score_difference'] for r in results),
        max_original_window_mrr_difference=max(r['max_original_window_mrr_difference'] for r in results),
        all_budget_timestamp_counts_verified=True, conditions=results))


if __name__ == '__main__':
    main()
