"""Derive reviewer diagnostics from frozen summaries, without training or score replay.

The optional original timestamp CSV is compacted once into the output directory.
Subsequent diagnostic runs can reuse that compact input.
"""
from argparse import ArgumentParser
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_composition(times, counts, size):
    """Reconstruct forward/inverse composition once test feedback fills the pool."""
    segments = deque()
    total = 0
    seen = 0
    rows = []
    for t, count in zip(times, counts):
        if seen >= size:
            assert total == size
            inverse = sum(n for _, direction, n in segments if direction == 'inverse')
            earliest = segments[0][0]
            earliest_retained = sum(n for when, _, n in segments if when == earliest)
            original_count = int(counts[np.searchsorted(times, earliest)])
            rows.append({
                'timestamp': int(t), 'window': size,
                'inverse_fraction': inverse / size,
                'span_index_units': int(t - earliest),
                'distinct_timestamps': len({when for when, _, _ in segments}),
                'boundary_retained_fraction': earliest_retained / original_count,
            })
        assert count % 2 == 0
        for direction in ('forward', 'inverse'):
            segments.append((int(t), direction, int(count // 2)))
        seen += int(count)
        total += int(count)
        while total > size:
            when, direction, n = segments.popleft()
            remove = min(total - size, n)
            total -= remove
            if n > remove:
                segments.appendleft((when, direction, n - remove))
    return rows


def parse_args(argv=None):
    ap = ArgumentParser()
    ap.add_argument('--frozen-root', type=Path, required=True,
                    help='Analysis root containing gate/, online/ and utility/.')
    ap.add_argument('--output-root', type=Path, required=True,
                    help='Diagnostic output directory; may contain previously compacted timestamp inputs.')
    ap.add_argument('--gate-timestamps', type=Path)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    frozen, out = args.frozen_root, args.output_root
    out.mkdir(parents=True, exist_ok=True)
    compact = out / 'gate_timestamp_inputs.csv.gz'
    source_record = out / 'timestamp_source.json'
    if args.gate_timestamps:
        g = pd.read_csv(args.gate_timestamps, float_precision='round_trip')
        g = g[g.score.isin(['softmax', 'minmax'])][[
            'case', 'seed', 'timestamp', 'score', 'query_count',
            'static_threshold', 'rolling_threshold', 'static_coverage', 'rolling_coverage',
        ]].copy()
        g.to_csv(compact, index=False, compression={'method': 'gzip', 'mtime': 0})
        source_record.write_text(json.dumps({
            'source_name': args.gate_timestamps.name,
            'source_sha256': sha(args.gate_timestamps),
            'compact_columns': list(g.columns),
            'scores_retained': ['minmax', 'softmax'],
            'float_parse': 'round_trip',
        }, indent=2) + '\n')
    g = pd.read_csv(compact, float_precision='round_trip')
    assert len(g) == 70160
    assert not g.duplicated(['case', 'seed', 'timestamp', 'score']).any()
    gate_seed = pd.read_csv(frozen / 'gate/gate_by_seed.csv')
    for (case, seed, score), d in g.groupby(['case', 'seed', 'score']):
        expected = gate_seed[(gate_seed.case == case) & (gate_seed.seed == seed) & (gate_seed.score == score)].iloc[0]
        for method in ['static', 'rolling']:
            actual = np.average(d[method + '_coverage'], weights=d.query_count)
            assert abs(actual - expected[method + '_coverage']) < 1e-12

    # The diagnostic is about logged thresholds, not unavailable candidate probabilities.
    s = g[g.score == 'softmax']
    soft_rows = []
    utility = pd.read_csv(frozen / 'utility/utility_summary.csv')
    for case, d in s.groupby('case'):
        u = utility[(utility.case == case) & (utility.method == 'label_softmax_rolling') & (utility.budget == 'unbounded')].iloc[0]
        soft_rows.append({
            'case': case, 'seed_timestamp_count': len(d),
            'static_threshold_equal_one_count': int((d.static_threshold == 1).sum()),
            'rolling_threshold_equal_one_count': int((d.rolling_threshold == 1).sum()),
            'minimum_static_probability_cutoff': float((1 - d.static_threshold).min()),
            'minimum_rolling_probability_cutoff': float((1 - d.rolling_threshold).min()),
            'rolling_full_vocabulary_rate': float(u.full_vocabulary_rate),
        })
    soft = pd.DataFrame(soft_rows)
    assert (soft.static_threshold_equal_one_count == 0).all()
    assert (soft.rolling_threshold_equal_one_count == 0).all()
    soft.to_csv(out / 'softmax_threshold_diagnostics.csv', index=False)

    # Query counts in the historical gate exporter are observed labels, both directions.
    m = g[g.score == 'minmax'].copy()
    m['dataset'] = np.where(m.case.str.startswith('icews14_'), 'icews14', 'icews05_15')
    assert m.groupby(['dataset', 'timestamp']).query_count.nunique().eq(1).all()
    window_rows = []
    for dataset, d in m.groupby('dataset'):
        counts = d.groupby('timestamp').query_count.first().sort_index()
        for window in [250, 500, 1000, 2000]:
            rows = window_composition(counts.index.to_numpy(), counts.to_numpy(), window)
            for row in rows:
                row['dataset'] = dataset
                window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    assert windows.inverse_fraction.between(.5, 1).all()
    windows.to_csv(out / 'window_composition_by_timestamp.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summary = windows.groupby(['dataset', 'window']).agg(
        eligible_timestamps=('timestamp', 'size'),
        inverse_fraction_mean=('inverse_fraction', 'mean'),
        inverse_fraction_min=('inverse_fraction', 'min'),
        inverse_fraction_max=('inverse_fraction', 'max'),
        span_index_units_mean=('span_index_units', 'mean'),
        distinct_timestamps_mean=('distinct_timestamps', 'mean'),
        boundary_retained_fraction_mean=('boundary_retained_fraction', 'mean'),
    ).reset_index()
    summary.to_csv(out / 'window_composition_summary.csv', index=False)

    # Existing summaries distinguish label weighting from equal-timestamp errors.
    online = pd.read_csv(frozen / 'online/by_seed.csv')
    online = online[online.score == 'minmax'].copy()
    cols = ['coverage', 'macro_time_absolute_target_error', 'macro_time_undercoverage', 'pool_span_mean']
    temporal = online.groupby(['case', 'method'])[cols].mean().reset_index()
    temporal.to_csv(out / 'temporal_error_summary.csv', index=False)
    seed_gate = gate_seed[gate_seed.score == 'minmax'].copy()
    seed_gate.to_csv(out / 'minmax_seed_effects.csv', index=False)

    # Ten chronological, nonoverlapping bins; methods share identical dates/labels.
    decile_rows = []
    for (case, seed), d in m.groupby(['case', 'seed']):
        d = d.sort_values('timestamp')
        for i, positions in enumerate(np.array_split(np.arange(len(d)), 10), 1):
            part = d.iloc[positions]
            for method in ['static', 'rolling']:
                decile_rows.append({
                    'case': case, 'seed': int(seed), 'decile': i, 'method': method,
                    'timestamp_first': int(part.timestamp.min()),
                    'timestamp_last': int(part.timestamp.max()),
                    'timestamp_count': len(part), 'label_count': int(part.query_count.sum()),
                    'coverage': np.average(part[method + '_coverage'], weights=part.query_count),
                })
    deciles = pd.DataFrame(decile_rows)
    assert len(deciles) == 1600
    deciles.to_csv(out / 'temporal_deciles_by_seed.csv', index=False)

    budgets = utility[(utility.method == 'label_minmax_rolling') & (utility.budget != 'unbounded')].copy()
    budgets['budget'] = budgets.budget.astype(int)
    budgets['budget_utilization'] = budgets.mean_set_size / budgets.budget
    assert budgets.budget_utilization.between(0, 1).all()
    budgets.to_csv(out / 'budget_actual_outputs_by_case.csv', index=False)
    budget_summary = budgets.groupby('budget').agg(
        cases=('case', 'size'), mean_returned_min=('mean_set_size', 'min'),
        mean_returned_max=('mean_set_size', 'max'),
        label_coverage_min=('label_coverage', 'min'), label_coverage_max=('label_coverage', 'max'),
        multi_coverage_min=('multi_full_set_coverage', 'min'), multi_coverage_max=('multi_full_set_coverage', 'max'),
    ).reset_index()
    assert budget_summary.cases.eq(16).all()
    budget_summary.to_csv(out / 'budget_actual_outputs_summary.csv', index=False)

    source_files = ['gate/gate_by_seed.csv', 'online/by_seed.csv', 'utility/utility_summary.csv']
    record = {
        'analysis': 'Post-review descriptive diagnostics from existing frozen outputs',
        'training_or_score_replay': False,
        'source_hashes': {name: sha(frozen / name) for name in source_files},
        'timestamp_source': json.loads(source_record.read_text()),
        'window_population': 'Test timestamps after test feedback alone fills each window; initial-calibration startup excluded. Forward labels precede inverse labels in each batch.',
        'softmax_scope': 'Immediate-feedback static and rolling-1000 gate thresholds only. No claim of candidate-level equivalence between float64 1-p and -p; delayed-feedback thresholds and raw probabilities were not re-evaluated.',
        'ranking_scope': 'No current-matrix MRR or pure Top-K baseline can be reconstructed from these aggregate inputs; no historical ranking results are substituted.',
        'resampling_interpretation': 'Empirical central 95% resampling sensitivity intervals on the observed output traces, not asserted frequentist confidence coverage for future streams.',
        'files': {str(p.relative_to(out)): sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'DIAGNOSTIC_MANIFEST.json'},
    }
    (out / 'DIAGNOSTIC_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'softmax_seed_timestamps': len(s), 'window_rows': len(windows), 'temporal_decile_rows': len(deciles), 'budget_cases': len(budgets), 'status': 'verified'}))


if __name__ == '__main__':
    main()
