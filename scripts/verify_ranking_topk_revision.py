"""Recompute ranking and candidate budgets locally from compact frozen arrays.

No torch, checkpoint, benchmark download or candidate-score replay is required.
Optionally supply --budget K --budget-output FILE for another output cap.
"""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True, help='Frozen ranking supplement directory.')
    ap.add_argument('--utility-root', type=Path, required=True, help='Directory containing utility_by_seed.csv.')
    ap.add_argument('--export-script', type=Path, default=ROOT / 'scripts/export_ranking_topk_revision.py',
                    help='Exporter whose hash is recorded in the freeze; pass the original script for historical evidence.')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--budget', type=int)
    ap.add_argument('--budget-output', type=Path)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    data = args.data_root
    if args.budget is not None:
        assert args.budget > 0 and args.budget_output
    freeze = json.loads((data / 'EVIDENCE_FREEZE.json').read_text())
    for name, expected in freeze['files'].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    assert freeze['script_sha256'] == hashlib.sha256(args.export_script.read_bytes()).hexdigest()
    verified = json.loads((data / 'verification.json').read_text())
    assert verified['full_matrix'] and verified['condition_count'] == 80
    expected_rank = pd.read_csv(data / 'ranking_by_seed.csv').set_index(['case', 'seed', 'direction'])
    expected_budget = pd.read_csv(data / 'topk_by_seed.csv').set_index(['case', 'seed', 'budget'])
    original_utility = pd.read_csv(args.utility_root / 'utility_by_seed.csv')
    original_utility = original_utility[original_utility.method == 'label_minmax_rolling'].set_index(['case', 'seed', 'budget'])
    hists = np.load(data / 'minmax_size_histograms.npz')
    groups = {}
    new_rows = []
    max_error = 0.
    labels = 0

    def same(a, b):
        nonlocal max_error
        error = abs(float(a) - float(b))
        max_error = max(max_error, error)
        assert error < 1e-10, (a, b)

    for condition in verified['conditions']:
        case, seed = condition['case'], condition['seed']
        fact_file = condition['fact_index_file']
        if fact_file not in groups:
            f = np.load(data / fact_file)
            facts = f['facts']
            assert hashlib.sha256(facts.astype('<i4').tobytes()).hexdigest() == condition['fact_array_sha256']
            _, inverse, counts = np.unique(facts[:, [0, 1, 3]], axis=0, return_inverse=True, return_counts=True)
            groups[fact_file] = (facts, inverse, counts, int(f['num_relations']))
        facts, inverse, counts, relations = groups[fact_file]
        ranks = np.load(data / 'ranks' / f'{case}_seed{seed}.npz')
        filtered, raw, eligible = ranks['filtered_rank'], ranks['raw_rank'], ranks['minmax_rolling_eligible']
        assert len(raw) == condition['label_count'] == len(facts)
        assert np.all(filtered > 0) and np.all(raw >= filtered)
        labels += len(raw)
        for direction, mask in [('both', np.ones(len(facts), bool)), ('tail', facts[:, 1] < relations), ('head', facts[:, 1] >= relations)]:
            r = expected_rank.loc[case, seed, direction]
            same(np.mean(1. / filtered[mask]), r.filtered_mrr)
            same(np.mean(1. / raw[mask]), r.raw_mrr)
            for k in (1, 3, 10):
                same(np.mean(filtered[mask] <= k), r[f'filtered_hits_at_{k}'])
        hist = hists[f'{case}_seed{seed}']
        assert hist.shape[0] == 3 and hist[0].sum() == len(counts)
        assert hist[1].sum() == np.sum(counts == 1) and hist[2].sum() == np.sum(counts > 1)
        vocab = hist.shape[1] - 1
        assert np.all(raw <= vocab)
        budgets = sorted(set([50, 100, 250, 500, 1000] + ([args.budget] if args.budget else [])))
        for budget in budgets:
            actual_cap = min(vocab, budget)
            row = dict(case=case, seed=seed, budget=budget, vocabulary_size=vocab)
            for method, covered in [('topk', raw <= actual_cap), ('minmax', (raw <= actual_cap) & eligible)]:
                recovered = np.bincount(inverse, weights=covered, minlength=len(counts))
                full = recovered == counts
                row.update({method + '_label_coverage': float(covered.mean()), method + '_full_set_coverage': float(full.mean()),
                    method + '_multi_full_set_coverage': float(full[counts > 1].mean()),
                    method + '_single_full_set_coverage': float(full[counts == 1].mean()),
                    method + '_partial_answer_recall': float(np.mean(recovered / counts)),
                    method + '_mean_set_size': float(actual_cap if method == 'topk' else np.dot(np.minimum(np.arange(vocab + 1), budget), hist[0]) / hist[0].sum())})
            row['ranking_exclusion'] = 1 - row['topk_label_coverage']
            row['threshold_exclusion'] = row['topk_label_coverage'] - row['minmax_label_coverage']
            row['candidate_saving_fraction'] = 1 - row['minmax_mean_set_size'] / actual_cap
            same(row['ranking_exclusion'] + row['threshold_exclusion'], 1 - row['minmax_label_coverage'])
            if budget in (50, 100, 250, 500, 1000):
                expected = expected_budget.loc[case, seed, budget]
                old = original_utility.loc[case, seed, str(budget)]
                for metric, value in row.items():
                    if metric not in ('case', 'seed', 'budget'):
                        same(value, expected[metric])
                    if metric.startswith('minmax_'):
                        same(value, old[metric.removeprefix('minmax_')])
            if budget == args.budget:
                new_rows.append(row)
    # Validate equal-seed means and sample standard deviations independently.
    for filename, source, keys, ignored in [('ranking_summary.csv', expected_rank.reset_index(), ['case', 'direction'], ['seed', 'label_count']), ('topk_summary.csv', expected_budget.reset_index(), ['case', 'budget'], ['seed', 'vocabulary_size'])]:
        expected = pd.read_csv(data / filename).set_index(keys)
        for key, frame in source.groupby(keys):
            assert len(frame) == 5
            for column in frame.select_dtypes('number').columns:
                if column not in keys + ignored:
                    same(frame[column].mean(), expected.loc[key, column + '_mean'])
                    same(frame[column].std(ddof=1), expected.loc[key, column + '_std'])
    if args.budget:
        args.budget_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(new_rows).sort_values(['case', 'seed']).to_csv(args.budget_output, index=False)
    report = dict(status='passed', conditions=80, ranking_rows=240, matched_budget_rows=400,
                  local_rank_labels_recomputed=labels, original_utility_seed_results_matched=True,
                  source_file_hashes_verified=len(freeze['files']), max_numeric_difference=max_error,
                  source_checkpoint_replay_facts=verified['replay_fact_count'], source_checkpoint_replay_timestamps=verified['replay_timestamp_count'],
                  source_checkpoint_score_difference=verified['max_true_score_difference'], new_budget=args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
