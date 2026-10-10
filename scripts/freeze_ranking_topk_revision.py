"""Attach verified size histograms and freeze a completed ranking supplement."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = ArgumentParser()
    ap.add_argument('--utility', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--export-script', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    verified = json.loads((out / 'verification.json').read_text())
    assert verified['full_matrix'] and verified['condition_count'] == 80
    histograms, sources = {}, {}
    for condition in verified['conditions']:
        key = f"{condition['case']}_seed{condition['seed']}"
        path = args.utility / 'conditions' / f'{key}.histograms.npz'
        marker = json.loads(path.with_name(f'{key}.complete.json').read_text())
        assert digest(path) == marker['outputs']['histograms']
        arrays = np.load(path)
        assert str(arrays['methods'][8]) == 'label_minmax_rolling'
        assert list(arrays['scopes']) == ['overall', 'single', 'multi']
        histograms[key] = arrays['histograms'][8]
        sources[str(path)] = digest(path)
    np.savez_compressed(out / 'minmax_size_histograms.npz', **histograms)
    (out / 'histogram_provenance.json').write_text(json.dumps(dict(method='label_minmax_rolling', method_index=8,
        scopes=['overall', 'single', 'multi'], histogram_bin='exact returned entity count before budget cap',
        sources_sha256=sources), indent=2) + '\n')
    manifest = dict(status='verified', conditions=80, script_sha256=digest(args.export_script),
        files={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'EVIDENCE_FREEZE.json'})
    (out / 'EVIDENCE_FREEZE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(manifest['files']), 'status': 'verified'}))


if __name__ == '__main__':
    main()
