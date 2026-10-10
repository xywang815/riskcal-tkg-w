"""Gate-only replay of frozen checkpoints, with resumable true-label scores.

No training, window selection, multi-answer evaluation, or utility experiments.
The primary contrast remains rolling-minus-static coverage at window 1000.
Minmax is invariant to positive affine score transforms; Softmax is not scale
invariant. All four encoders and both scorer/dataset families are retained.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from riskcal_tkg.artifacts import load_verified_checkpoint, sha256_file
from riskcal_tkg.calibration import CalibrationPool, finite_sample_quantile
from riskcal_tkg.config import load_config
from riskcal_tkg.data import add_inverse_relations, load_configured_table, split_calibration_roles, temporal_split
from riskcal_tkg.experiment import _config_hash_payload, _dataset_manifest, _jsonable, _score_all_objects
from riskcal_tkg.model import build_temporal_model
from scripts.export_time_encoder_audit import EXPECTED_RUNS, bootstrap_statistic

SCORES = ('margin', 'negscore', 'minmax', 'softmax')

def true_nonconformity(scores, labels, temperature=1.0):
    """Float64 definitions matching the training evaluator without NxE masks."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('candidate scores must be a finite matrix')
    if labels.shape != (len(values),) or np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError('invalid true labels')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive')
    actual = values[np.arange(len(values)), labels]
    minima, maxima = values.min(axis=1), values.max(axis=1)
    ranges = maxima - minima
    scaled = np.divide(actual-minima, ranges, out=np.zeros_like(actual), where=ranges > 0)
    shifted = (values - maxima[:, None]) / temperature
    probability = np.exp((actual-maxima)/temperature) / np.exp(shifted).sum(axis=1)
    return {'margin':maxima-actual, 'negscore':-actual, 'minmax':-scaled, 'softmax':1-probability}

def replay_coverage(calibration, test, window=1000, target=0.9):
    if calibration.timestamp.max() >= test.timestamp.min():
        raise ValueError('calibration and test timestamps overlap')
    pools = {s:CalibrationPool(window) for s in SCORES}
    static = {s:finite_sample_quantile(calibration[s].to_numpy(), 1-target) for s in SCORES}
    for timestamp, group in calibration.groupby('timestamp', sort=True):
        for s in SCORES:
            pools[s].add(int(timestamp), group[s].to_numpy())
    rows = []
    for timestamp, group in test.groupby('timestamp', sort=True):
        for s in SCORES:
            past = pools[s].values_before(int(timestamp))
            threshold = finite_sample_quantile(past.values, 1-target)
            scores = group[s].to_numpy()
            cov_s = float(np.mean(scores <= static[s]))
            cov_r = float(np.mean(scores <= threshold))
            rows.append({'timestamp':int(timestamp), 'score':s, 'query_count':len(group),
                         'static_coverage':cov_s, 'rolling_coverage':cov_r,
                         'rolling_minus_static_coverage':cov_r-cov_s,
                         'static_threshold':static[s], 'rolling_threshold':threshold,
                         'calibration_max_timestamp':int(past.timestamps.max())})
        # Commit feedback only after every prediction at the timestamp.
        for s in SCORES:
            pools[s].add(int(timestamp), group[s].to_numpy())
    return pd.DataFrame(rows)

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)

def source_identity(repo):
    paths = [Path(__file__), *sorted((repo/'src/riskcal_tkg').glob('*.py')), repo/'scripts/export_time_encoder_audit.py']
    return {str(p.relative_to(repo)):sha256_file(p) for p in paths}

def score_split(model, values, relation_count, config, phase):
    frames = []
    for timestamp in np.unique(values[:,3]):
        facts = add_inverse_relations(values[values[:,3] == timestamp], relation_count)
        scores = _score_all_objects(model, facts, config.batch_size)
        columns = true_nonconformity(scores, facts[:,2], config.kgcp_temperature)
        frame = pd.DataFrame(columns)
        for index, name in enumerate(('subject_id','relation_id','true_object_id','timestamp')):
            frame[name] = facts[:,index]
        frame['phase'] = phase
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def verify_replay(rows, original, seed):
    comparisons = []
    for score in SCORES:
        methods = [('static_coverage','static_margin' if score == 'margin' else f'kgcp_{score}_static')]
        if score == 'margin':
            methods.append(('rolling_coverage','rolling_margin'))
        for col, method in methods:
            old = original[(original.seed == seed) & (original.method == method)]
            new = rows[rows.score == score]
            merged = new.merge(old[['timestamp','coverage','query_count']], on='timestamp', suffixes=('','_old'), validate='one_to_one')
            if len(merged) != len(old) or len(merged) != len(new) or not (merged.query_count == merged.query_count_old).all():
                raise ValueError('replay row identity mismatch')
            delta = float(np.max(np.abs(merged[col]-merged.coverage)))
            # CSV roundoff is permitted; a single changed inclusion is not.
            if delta > 1e-10:
                raise ValueError(f'replay mismatch for {method}: maximum coverage difference {delta}')
            comparisons.append({'method':method,'max_coverage_difference':delta})
    return comparisons

def main():
    ap = ArgumentParser()
    ap.add_argument('--matrix-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--case', action='append', choices=EXPECTED_RUNS)
    ap.add_argument('--seed', action='append', type=int, choices=(17,29,43,59,71))
    ap.add_argument('--skip-bootstrap', action='store_true')
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = a.output_root.resolve(); out.mkdir(parents=True,exist_ok=True)
    progress = json.loads((a.matrix_root/'matrix_progress.json').read_text())
    sources = source_identity(repo)
    specification = {'cases':list(a.case or EXPECTED_RUNS), 'seeds':a.seed or [17,29,43,59,71], 'scores':list(SCORES), 'window':1000,'target':0.9,'evaluation_device':'cpu (same as original evaluator)','training_git_commit':progress['git_commit'],'sources':sources}
    spec_hash = hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    if (out/'gate_specification.json').exists():
        if json.loads((out/'gate_specification.json').read_text()) != specification:
            raise ValueError('existing analysis uses a different frozen specification/source')
    else:
        atomic_json(out/'gate_specification.json', specification)
    torch.set_num_threads(4)
    data_cache = {}; frames = []; records = []
    for case in specification['cases']:
        rec = progress['runs'][case]
        if rec['status'] != 'complete': raise ValueError(f'incomplete source {case}')
        root = Path(rec['run_root']); cfg_path = Path(rec['config']); cfg = load_config(cfg_path)
        if sha256_file(cfg_path) != rec['config_sha256']: raise ValueError('config file identity changed')
        config_hash = hashlib.sha256(json.dumps(_config_hash_payload(_jsonable(asdict(cfg))),sort_keys=True).encode()).hexdigest()
        if cfg.rolling_window != 1000 or cfg.target_coverage != 0.9: raise ValueError('protocol mismatch')
        if cfg.data_mode not in data_cache:
            table = load_configured_table(cfg)
            split = temporal_split(table,train_fraction=cfg.train_fraction,calibration_fraction=cfg.calibration_fraction)
            roles = split_calibration_roles(split.calibration,fractions=cfg.calibration_role_fractions)
            data_cache[cfg.data_mode] = (table, split, roles, _dataset_manifest(table,split,cfg)['sha256'])
        table, split, roles, dataset_hash = data_cache[cfg.data_mode]
        original = pd.read_csv(root/'metrics/per_window.csv')
        for seed in specification['seeds']:
            started = time.monotonic(); label = f'seed{seed}_delete0p00'; key = f'{case}_seed{seed}'
            marker = json.loads((root/'conditions'/f'{label}.complete.json').read_text())
            checkpoint_path = root/'checkpoints'/f'{label}.ckpt'
            checkpoint_hash = sha256_file(checkpoint_path)
            if marker['status'] != 'complete' or checkpoint_hash != marker['artifacts']['checkpoint']: raise ValueError('checkpoint checksum mismatch')
            if marker['config_sha256'] != config_hash or marker['dataset_sha256'] != dataset_hash: raise ValueError('checkpoint/data/config identity mismatch')
            marker_out = out/'conditions'/f'{key}.complete.json'
            csv_out = out/'conditions'/f'{key}.csv'; scores_out = out/'scores'/f'{key}.parquet'
            if marker_out.exists():
                saved = json.loads(marker_out.read_text())
                if saved['specification_sha256'] != spec_hash or saved['checkpoint_sha256'] != checkpoint_hash or saved['output_sha256'] != sha256_file(csv_out) or saved['scores_sha256'] != sha256_file(scores_out): raise ValueError('resume artifact mismatch')
                frames.append(pd.read_csv(csv_out)); records.append(saved)
                print(f'[gate] verified {key}',flush=True); continue
            print(f'[gate] starting {key}',flush=True)
            payload = load_verified_checkpoint(checkpoint_path,config_sha256=config_hash,dataset_sha256=dataset_hash,deletion_mask_sha256=marker['deletion_mask_sha256'])
            model = build_temporal_model(cfg.model_name,len(table.entity_to_id),2*len(table.relation_to_id),len(table.timestamp_to_id),cfg.embedding_dim,time_scale=int(float(payload['time_scale'])),time_encoding=cfg.time_encoding)
            model.load_state_dict(payload['state_dict'],strict=True); model = model.to('cpu')
            cal = score_split(model,roles.final_calibration.values,len(table.relation_to_id),cfg,'calibration')
            test = score_split(model,split.test.values,len(table.relation_to_id),cfg,'test')
            rows = replay_coverage(cal,test)
            comparisons = verify_replay(rows,original,seed)
            for name,value in {'case':case,'seed':seed,'dataset_mode':cfg.data_mode,'model_name':cfg.model_name,'time_encoding':cfg.time_encoding}.items(): rows[name] = value
            scores_out.parent.mkdir(parents=True,exist_ok=True); csv_out.parent.mkdir(parents=True,exist_ok=True)
            tmp = scores_out.with_suffix('.parquet.tmp'); pd.concat([cal,test],ignore_index=True).to_parquet(tmp,index=False); tmp.replace(scores_out)
            tmp = csv_out.with_suffix('.csv.tmp'); rows.to_csv(tmp,index=False); tmp.replace(csv_out)
            record = {'status':'complete','case':case,'seed':seed,'checkpoint_sha256':checkpoint_hash,'dataset_sha256':dataset_hash,'config_sha256':config_hash,'specification_sha256':spec_hash,'output_sha256':sha256_file(csv_out),'scores_sha256':sha256_file(scores_out),'comparisons':comparisons,'seconds':time.monotonic()-started}
            atomic_json(marker_out,record); frames.append(rows); records.append(record)
            print(f'[gate] completed {key}: {record["seconds"]:.1f}s ({len(records)}/{len(specification["cases"])*len(specification["seeds"])})',flush=True)
            del model,payload,cal,test; torch.cuda.empty_cache()
    all_rows = pd.concat(frames,ignore_index=True); all_rows.to_csv(out/'gate_by_timestamp.csv',index=False)
    summaries = []
    for (case,score,seed), g in all_rows.groupby(['case','score','seed'],sort=True):
        row = {'case':case,'score':score,'seed':int(seed)}
        for col in ('static_coverage','rolling_coverage','rolling_minus_static_coverage'):
            row[col] = float(np.average(g[col],weights=g.query_count))
        row['aggregate_target_error_reduction'] = abs(row['static_coverage']-0.9)-abs(row['rolling_coverage']-0.9)
        summaries.append(row)
    by_seed = pd.DataFrame(summaries); by_seed.to_csv(out/'gate_by_seed.csv',index=False)
    summary = by_seed.groupby(['case','score'],as_index=False).mean(numeric_only=True).drop(columns='seed')
    bootstrap = []
    if not a.skip_bootstrap:
        for index, ((case,score),g) in enumerate(all_rows.groupby(['case','score'],sort=True)):
            for block in (3,7,14,21):
                result = bootstrap_statistic(g,'rolling_minus_static_coverage',block_length=block,iterations=20000,bootstrap_seed=20260904+1000*index+block)
                result['bootstrap_nonpositive_fraction'] = result.pop('pvalue_positive')
                bootstrap.append({'case':case,'score':score,'block_length':block,**result})
        b = pd.DataFrame(bootstrap); b.to_csv(out/'gate_bootstrap.csv',index=False)
        summary = summary.merge(b[b.block_length == 7],on=['case','score'],validate='one_to_one')
    summary.to_csv(out/'gate_summary.csv',index=False)
    status = 'complete' if not a.skip_bootstrap else 'point_estimates_only'
    atomic_json(out/'gate_manifest.json',{'status':status,'created_at':datetime.now(UTC).isoformat(),'condition_count':len(records),'specification':specification,'specification_sha256':spec_hash,'iterations':0 if a.skip_bootstrap else 20000,'block_lengths':[3,7,14,21],'provenance':records,'interval_note':'Pointwise paired equal-seed circular moving-block intervals; not simultaneous intervals or a calibrated family of hypothesis tests. Seed repeats are not independent populations.'})
    inventory = [f'{sha256_file(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.txt']
    (out/'SHA256SUMS.txt').write_text('\n'.join(inventory)+'\n')
    print('GATE_EXPORT_COMPLETE',out,flush=True)

if __name__ == '__main__':
    main()
