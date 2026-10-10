"""Resumable online/feedback stages from the verified P0 score cache only."""
from argparse import ArgumentParser
from datetime import datetime, UTC
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from riskcal_tkg.calibration import CalibrationPool, finite_sample_quantile
from scripts.export_window_ablation import AdaptiveAlphaState
from scripts.cached_followup_bootstrap import bootstrap_many
from scripts.export_time_encoder_audit import EXPECTED_RUNS

SCORES=('margin','negscore','minmax','softmax')
SEEDS=(17,29,43,59,71)
WINDOWS=(250,500,1000,2000)
HALF_LIVES=(7.,14.,30.,math.inf)
GAMMAS=(.001,.005,.01,.05)
DELAYS=(0,1,3,7,14)
RETENTIONS=(1.,.9,.7)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def sorted_quantile(values,alpha):
    if len(values)==0 or not 0<alpha<1: raise ValueError('invalid quantile inputs')
    return float(values[min(len(values),math.ceil((len(values)+1)*(1-alpha)))-1])

class SortedHistory:
    def __init__(self,values,timestamps):
        order=np.argsort(values,kind='stable')
        self.values=np.asarray(values,dtype=float)[order]
        self.timestamps=np.asarray(timestamps,dtype=int)[order]
    def add(self,values,timestamp):
        incoming=np.sort(np.asarray(values,dtype=float))
        if not len(incoming): return
        positions=np.searchsorted(self.values,incoming,side='right')+np.arange(len(incoming))
        remain=np.ones(len(self.values)+len(incoming),dtype=bool);remain[positions]=False
        joined=np.empty(len(remain));joined[positions]=incoming;joined[remain]=self.values
        times=np.empty(len(remain),dtype=int);times[positions]=timestamp;times[remain]=self.timestamps
        self.values,self.timestamps=joined,times
    def weighted_quantile(self,timestamp,half_life,target=.9):
        if (self.timestamps>=timestamp).any(): raise ValueError('future feedback in weighted history')
        weights=np.ones(len(self.values)) if math.isinf(half_life) else np.exp2(-(timestamp-self.timestamps)/half_life)
        cumulative=np.cumsum(weights)/weights.sum()
        return float(self.values[min(np.searchsorted(cumulative,target,side='left'),len(self.values)-1)])

def _row(timestamp,score,method,values,threshold,pool,span,alpha=.1):
    return {'timestamp':int(timestamp),'score':score,'method':method,'query_count':len(values),
            'coverage':float(np.mean(values<=threshold)),'threshold':float(threshold),
            'pool_score_count':int(pool),'pool_span_timestamps':int(span),'aci_alpha':float(alpha)}

def online_rows(cal,test):
    if cal.timestamp.max()>=test.timestamp.min(): raise ValueError('overlapping calibration/test')
    histories={s:SortedHistory(cal[s].to_numpy(),cal.timestamp.to_numpy()) for s in SCORES}
    recent={s:CalibrationPool(max(WINDOWS)) for s in SCORES}
    static={s:finite_sample_quantile(cal[s].to_numpy(),.1) for s in SCORES}
    aci={(s,g):AdaptiveAlphaState(.1,g,.1) for s in SCORES for g in GAMMAS}
    for t,batch in cal.groupby('timestamp',sort=True):
        for s in SCORES: recent[s].add(t,batch[s].to_numpy())
    rows=[]
    for t,batch in test.groupby('timestamp',sort=True):
        t=int(t)
        for s in SCORES:
            v=batch[s].to_numpy();hist=histories[s]
            if (hist.timestamps>=t).any(): raise ValueError('future labels entered history')
            last=recent[s].values_before(t)
            rows.append(_row(t,s,'static',v,static[s],len(cal),t-cal.timestamp.min()))
            rows.append(_row(t,s,'expanding',v,sorted_quantile(hist.values,.1),len(hist.values),t-hist.timestamps.min()))
            for window in WINDOWS:
                pool=last.values[-window:];times=last.timestamps[-window:]
                rows.append(_row(t,s,f'rolling_{window}',v,finite_sample_quantile(pool,.1),len(pool),t-times.min()))
            for half in HALF_LIVES:
                rows.append(_row(t,s,f'weighted_expanding_h{half:g}',v,hist.weighted_quantile(t,half),len(hist.values),t-hist.timestamps.min()))
            for gamma in GAMMAS:
                state=aci[s,gamma]
                row=_row(t,s,f'aci_gamma_{gamma:g}',v,sorted_quantile(hist.values,state.alpha),len(hist.values),t-hist.timestamps.min(),state.alpha)
                rows.append(row);state.update(1-row['coverage'])
        for s in SCORES:
            v=batch[s].to_numpy();histories[s].add(v,t);recent[s].add(t,v)
    return pd.DataFrame(rows)

def feedback_rows(cal,test,seed):
    if cal.timestamp.max()>=test.timestamp.min(): raise ValueError('overlapping calibration/test')
    policies=[(d,r) for d in DELAYS for r in RETENTIONS]
    histories={(d,r,s):CalibrationPool(1000) for d,r in policies for s in SCORES}
    pending={p:[] for p in policies}
    rngs={p:np.random.default_rng(20260904+100000*int(seed)+1000*p[0]+round(100*p[1])) for p in policies}
    static={s:finite_sample_quantile(cal[s].to_numpy(),.1) for s in SCORES}
    for t,batch in cal.groupby('timestamp',sort=True):
        for d,r in policies:
            for s in SCORES: histories[d,r,s].add(t,batch[s].to_numpy())
    rows=[]
    for index,(t,batch) in enumerate(test.groupby('timestamp',sort=True)):
        t=int(t)
        for d,r in policies:
            keep_pending=[]
            for due,when,values in pending[d,r]:
                if due<=index:
                    for j,s in enumerate(SCORES): histories[d,r,s].add(when,values[:,j])
                else: keep_pending.append((due,when,values))
            pending[d,r]=keep_pending
        for s in SCORES:
            current=batch[s].to_numpy()
            rows.append(_row(t,s,'static',current,static[s],len(cal),t-cal.timestamp.min()))
            for d,r in policies:
                history=histories[d,r,s].values_before(t)
                row=_row(t,s,f'rolling_delay{d}_retain{r:g}',current,finite_sample_quantile(history.values,.1),len(history.values),t-history.timestamps.min())
                row.update(extra_delay_blocks=d,feedback_retention=r);rows.append(row)
        # Draw one feedback mask per policy, shared by every score definition.
        current=batch[list(SCORES)].to_numpy()
        for d,r in policies:
            keep=rngs[d,r].random(len(current))<r
            if keep.any(): pending[d,r].append((index+d+1,t,current[keep]))
    return pd.DataFrame(rows)

def verify_against_gate(rows,gate,phase):
    method='rolling_1000' if phase=='online' else 'rolling_delay0_retain1'
    for new_method,old_col in [('static','static_coverage'),(method,'rolling_coverage')]:
        left=rows[rows.method==new_method]
        merged=left.merge(gate[['timestamp','score','query_count',old_col]],on=['timestamp','score'],suffixes=('','_gate'),validate='one_to_one')
        if len(merged)!=len(gate) or len(left)!=len(gate) or not (merged.query_count==merged.query_count_gate).all(): raise ValueError('cached replay row mismatch')
        if np.max(np.abs(merged.coverage-merged[old_col]))>1e-10: raise ValueError('cached replay disagrees with gate')

def summarize_and_bootstrap(rows,out,phase):
    per_seed=[]
    for keys,g in rows.groupby(['case','score','method','seed'],sort=True):
        coverage=float(np.average(g.coverage,weights=g.query_count))
        per_seed.append(dict(zip(['case','score','method','seed'],keys),coverage=coverage,
            aggregate_target_error=abs(coverage-.9),macro_time_undercoverage=float(np.maximum(.9-g.coverage,0).mean()),
            macro_time_absolute_target_error=float(np.abs(.9-g.coverage).mean()),
            pool_count_mean=float(g.pool_score_count.mean()),pool_span_mean=float(g.pool_span_timestamps.mean())))
    by_seed=pd.DataFrame(per_seed);by_seed.to_csv(out/'by_seed.csv',index=False)
    summary=by_seed.groupby(['case','score','method'],as_index=False).mean(numeric_only=True).drop(columns='seed')
    reference='static' if phase=='online' else 'rolling_delay0_retain1'
    boots=[]
    for index,((case,score),g) in enumerate(rows.groupby(['case','score'],sort=True)):
        methods=sorted(g.method.unique());seeds=sorted(g.seed.unique());times=sorted(g.timestamp.unique())
        grid=pd.MultiIndex.from_product([seeds,times],names=['seed','timestamp'])
        wide=g.pivot(index=['seed','timestamp'],columns='method',values='coverage').reindex(grid)
        if wide.isna().any().any(): raise ValueError('missing seed/timestamp/method cells')
        weights=g.groupby(['seed','timestamp']).query_count.first().reindex(grid).to_numpy().reshape(len(seeds),len(times))
        values=(wide[methods].to_numpy()-wide[[reference]].to_numpy()).reshape(len(seeds),len(times),len(methods))
        for block in (3,7,14,21):
            result=bootstrap_many(values,weights,block_length=block,iterations=20000,bootstrap_seed=20260908+1000*index+block)
            for j,method in enumerate(methods):
                boots.append({'case':case,'score':score,'method':method,'reference':reference,'block_length':block,
                              **{name:float(v[j]) for name,v in result.items()}})
        print(f'[bootstrap] {phase} {case} {score}',flush=True)
    bootstrap=pd.DataFrame(boots);bootstrap.to_csv(out/'bootstrap.csv',index=False)
    summary=summary.merge(bootstrap[bootstrap.block_length==7],on=['case','score','method'],validate='one_to_one')
    summary.to_csv(out/'summary.csv',index=False)

def main():
    ap=ArgumentParser()
    ap.add_argument('--phase',choices=('online','feedback'),required=True)
    ap.add_argument('--gate-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--online-root',type=Path)
    ap.add_argument('--analysis-git-commit',required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    if len(a.analysis_git_commit)!=40 or any(c not in '0123456789abcdef' for c in a.analysis_git_commit): raise ValueError('full analysis Git commit required')
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((a.gate_root/'gate_manifest.json').read_text())
    if gate['status']!='complete' or gate['condition_count']!=80: raise ValueError('complete P0 gate required')
    if a.phase=='feedback':
        if a.online_root is None: raise ValueError('online stage required first')
        online=json.loads((a.online_root/'stage_manifest.json').read_text())
        if online['status']!='complete' or online['phase']!='online' or online['condition_count']!=80: raise ValueError('online stage incomplete')
    source_paths=[Path(__file__),repo/'scripts/cached_followup_bootstrap.py',repo/'scripts/export_window_ablation.py',repo/'src/riskcal_tkg/calibration.py',repo/'README.md']
    spec={'phase':a.phase,'cases':list(EXPECTED_RUNS),'seeds':list(SEEDS),'scores':list(SCORES),'analysis_git_commit':a.analysis_git_commit,
          'windows':list(WINDOWS),'half_lives':['7','14','30','inf'],'aci_gammas':list(GAMMAS),
          'delays':list(DELAYS),'retentions':list(RETENTIONS),'feedback_seed':20260904,
          'target':.9,'iterations':20000,'block_lengths':[3,7,14,21],
          'gate_manifest_sha256':sha(a.gate_root/'gate_manifest.json'),
          'online_manifest_sha256':None if a.online_root is None else sha(a.online_root/'stage_manifest.json'),
          'sources':{str(p.relative_to(repo)):sha(p) for p in source_paths}}
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if (out/'specification.json').exists() and json.loads((out/'specification.json').read_text())!=spec: raise ValueError('changed stage specification')
    write_json(out/'specification.json',spec)
    frames=[];records=[]
    for case in EXPECTED_RUNS:
        for seed in SEEDS:
            key=f'{case}_seed{seed}';started=time.monotonic()
            source=a.gate_root/'scores'/f'{key}.parquet';source_hash=sha(source)
            source_marker=json.loads((a.gate_root/'conditions'/f'{key}.complete.json').read_text())
            if source_hash!=source_marker['scores_sha256']: raise ValueError('source cache checksum mismatch')
            gate_csv=a.gate_root/'conditions'/f'{key}.csv'
            if sha(gate_csv)!=source_marker['output_sha256']: raise ValueError('source gate rows checksum mismatch')
            marker_path=out/'conditions'/f'{key}.complete.json';csv_path=out/'conditions'/f'{key}.csv'
            if marker_path.exists():
                record=json.loads(marker_path.read_text())
                if record['specification_sha256']!=spec_hash or record['source_sha256']!=source_hash or record['output_sha256']!=sha(csv_path): raise ValueError('resume checksum mismatch')
                frame=pd.read_csv(csv_path);print(f'[stage] verified {key}',flush=True)
            else:
                print(f'[stage] {a.phase} starting {key}',flush=True)
                data=pd.read_parquet(source);cal=data[data.phase=='calibration'];test=data[data.phase=='test']
                frame=online_rows(cal,test) if a.phase=='online' else feedback_rows(cal,test,seed)
                frame['case']=case;frame['seed']=seed
                verify_against_gate(frame,pd.read_csv(gate_csv),a.phase)
                if frame.duplicated(['score','timestamp','method']).any(): raise ValueError('duplicate rows')
                csv_path.parent.mkdir(parents=True,exist_ok=True)
                tmp=csv_path.with_suffix('.csv.tmp');frame.to_csv(tmp,index=False);tmp.replace(csv_path)
                record={'status':'complete','case':case,'seed':seed,'specification_sha256':spec_hash,'source_sha256':source_hash,'output_sha256':sha(csv_path),'row_count':len(frame),'seconds':time.monotonic()-started}
                write_json(marker_path,record)
                print(f'[stage] {a.phase} completed {key}: {record["seconds"]:.1f}s',flush=True)
            frames.append(frame);records.append(record)
    rows=pd.concat(frames,ignore_index=True);rows.to_csv(out/'by_timestamp.csv',index=False)
    summarize_and_bootstrap(rows,out,a.phase)
    write_json(out/'stage_manifest.json',{'status':'complete','phase':a.phase,'created_at':datetime.now(UTC).isoformat(),'condition_count':80,'row_count':len(rows),'specification':spec,'specification_sha256':spec_hash,'provenance':records,'inference_note':'Paired equal-seed circular moving-block, pointwise intervals. Seed repeats are not independent populations; nonpositive fractions are not calibrated p-values. No utility inferred from label scores.'})
    inventory=[f'{sha(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt']
    (out/'SHA256SUMS.txt').write_text('\n'.join(inventory)+'\n')
    print('STAGE_COMPLETE',a.phase,out,flush=True)

if __name__=='__main__': main()
