from pathlib import Path

import pytest

from scripts import analyze_review_diagnostics as diagnostics
from scripts import export_ranking_topk_revision as ranking
from scripts import verify_ranking_topk_revision as verifier


def test_ranking_accepts_independent_external_input_roots():
    args = ranking.parse_args([
        '--matrix-root', '/external/training',
        '--gate-root', '/external/gate',
        '--utility-root', '/external/utility',
        '--output', '/new/ranking',
    ])
    assert args.matrix_root == Path('/external/training')
    assert args.gate_root == Path('/external/gate')
    assert args.utility_root == Path('/external/utility')
    assert args.base is None


def test_ranking_preserves_legacy_base_and_allows_explicit_override():
    args = ranking.parse_args([
        '--base', '/legacy', '--gate-root', '/relocated/gate',
        '--output', '/new/ranking',
    ])
    assert args.matrix_root == Path('/legacy/time_encoder_audit_matrix')
    assert args.gate_root == Path('/relocated/gate')
    assert args.utility_root == Path('/legacy/analysis/utility_0690baf')


@pytest.mark.parametrize('missing', ['matrix-root', 'gate-root', 'utility-root'])
def test_ranking_requires_each_input_without_legacy_base(missing, capsys):
    argv = ['--output', '/new/ranking']
    for name in ('matrix-root', 'gate-root', 'utility-root'):
        if name != missing:
            argv.extend(['--' + name, '/external/' + name])
    with pytest.raises(SystemExit) as exc:
        ranking.parse_args(argv)
    assert exc.value.code == 2
    assert '--' + missing in capsys.readouterr().err


def test_verifier_accepts_external_evidence_and_historical_exporter():
    args = verifier.parse_args([
        '--data-root', '/evidence/ranking',
        '--utility-root', '/evidence/utility',
        '--export-script', '/original/export_ranking_topk_revision.py',
        '--output', '/new/verification.json',
        '--budget', '750', '--budget-output', '/new/top750.csv',
    ])
    assert args.data_root == Path('/evidence/ranking')
    assert args.utility_root == Path('/evidence/utility')
    assert args.export_script == Path('/original/export_ranking_topk_revision.py')
    assert args.budget == 750


def test_verifier_defaults_to_current_exporter():
    args = verifier.parse_args([
        '--data-root', '/evidence/ranking', '--utility-root', '/evidence/utility',
        '--output', '/new/verification.json',
    ])
    assert args.export_script == Path(ranking.__file__).resolve()


def test_diagnostics_separates_external_inputs_from_new_outputs():
    args = diagnostics.parse_args([
        '--frozen-root', '/evidence/analysis', '--output-root', '/new/diagnostics',
        '--gate-timestamps', '/evidence/gate_timestamp_inputs.csv.gz',
    ])
    assert args.frozen_root == Path('/evidence/analysis')
    assert args.output_root == Path('/new/diagnostics')
    assert args.gate_timestamps == Path('/evidence/gate_timestamp_inputs.csv.gz')
