"""Full-matrix, resumable multi-answer utility from frozen scorer checkpoints."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime,UTC
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib
import json
import multiprocessing
import platform
import time

import numpy as np
import pandas as pd
import torch

from riskcal_tkg.artifacts import load_verified_checkpoint,sha256_file
from riskcal_tkg.calibration import CalibrationPool,finite_sample_quantile,_kgcp_candidate_nonconformity
from riskcal_tkg.config import load_config
from riskcal_tkg.experiment import _score_all_objects,_config_hash_payload,_jsonable
from riskcal_tkg.followup import build_query_grouping
from riskcal_tkg.model import build_temporal_model
from riskcal_tkg.utility_frontier import SUM_FIELDS,candidate_order,prefix_sizes,query_sufficient,summarize_sufficient,histogram_quantile,clipped_histogram
from scripts.export_cached_calibration_followups import SortedHistory,sorted_quantile
from scripts.export_time_encoder_audit import EXPECTED_RUNS
from scripts.utility_bootstrap import bootstrap_ratios

SCORES=('margin','negscore','minmax','softmax')
HISTORIES=('static','expanding','rolling')
METHODS=tuple(f'label_{s}_{h}' for s in SCORES for h in HISTORIES)+tuple(f'query_max_margin_{h}' for h in HISTORIES)
BUDGETS=('unbounded','50','100','250','500','1000')
SEEDS=(17,29,43,59,71)
FACT_COLUMNS=['subject_id','relation_id','true_object_id','timestamp']
RATIOS={
 'label_coverage':('covered_label_count','label_count'),
 'full_set_coverage':('full_query_count','query_count'),
 'single_full_set_coverage':('single_full_query_count','single_query_count'),
 'multi_full_set_coverage':('multi_full_query_count','multi_query_count'),
 'mean_set_size':('set_size_sum','query_count'),
 'single_mean_set_size':('single_set_size_sum','single_query_count'),
 'multi_mean_set_size':('multi_set_size_sum','multi_query_count'),
 'partial_answer_recall':('partial_recall_sum','query_count'),
 'full_vocabulary_rate':('full_vocabulary_query_count','query_count'),
}

def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def prefix(budget): return '' if budget=='unbounded' else f'budget_{budget}_'

def _query_max(frame):
    group=build_query_grouping(frame[FACT_COLUMNS].to_numpy(dtype=np.int64))
    result=np.full(group.query_count,-np.inf)
    np.maximum.at(result,group.inverse,frame.margin.to_numpy())
    return result

def evaluate_condition(case,seed,record,gate_root):
    started=time.monotonic();root=Path(record['run_root']);key=f'{case}_seed{seed}'
    cfg_path=Path(record['config']);cfg=load_config(cfg_path)
    if sha256_file(cfg_path)!=record['config_sha256']: raise ValueError('source config changed')
    config_hash=hashlib.sha256(json.dumps(_config_hash_payload(_jsonable(asdict(cfg))),sort_keys=True).encode()).hexdigest()
    marker_path=root/'conditions'/f'seed{seed}_delete0p00.complete.json'
    marker=json.loads(marker_path.read_text());dataset=json.loads((root/'dataset_manifest.json').read_text())
    checkpoint=root/'checkpoints'/f'seed{seed}_delete0p00.ckpt'
    checkpoint_hash=sha256_file(checkpoint)
    if marker['status']!='complete' or checkpoint_hash!=marker['artifacts']['checkpoint']: raise ValueError('unverified checkpoint')
    if marker['config_sha256']!=config_hash or marker['dataset_sha256']!=dataset['sha256']: raise ValueError('checkpoint identity mismatch')
    source_marker=json.loads((gate_root/'conditions'/f'{key}.complete.json').read_text())
    cache=gate_root/'scores'/f'{key}.parquet';gate_csv=gate_root/'conditions'/f'{key}.csv'
    if sha256_file(cache)!=source_marker['scores_sha256'] or sha256_file(gate_csv)!=source_marker['output_sha256']: raise ValueError('P0 cache checksum mismatch')
    payload=load_verified_checkpoint(checkpoint,config_sha256=config_hash,dataset_sha256=dataset['sha256'],deletion_mask_sha256=marker['deletion_mask_sha256'])
    vocabulary=int(dataset['num_entities']);relations=int(dataset['num_relations'])
    model=build_temporal_model(cfg.model_name,vocabulary,2*relations,int(dataset['num_timestamps']),cfg.embedding_dim,time_scale=int(float(payload['time_scale'])),time_encoding=cfg.time_encoding)
    model.load_state_dict(payload['state_dict'],strict=True);model=model.to('cpu')
    cache_rows=pd.read_parquet(cache);cal=cache_rows[cache_rows.phase=='calibration'];test=cache_rows[cache_rows.phase=='test']
    if cal.timestamp.max()>=test.timestamp.min(): raise ValueError('non-past initial calibration')
    values={s:[] for s in (*SCORES,'query_max_margin')}
    times={s:[] for s in values};recent={s:CalibrationPool(1000) for s in values}
    for timestamp,frame in cal.groupby('timestamp',sort=True):
        batch={s:frame[s].to_numpy() for s in SCORES};batch['query_max_margin']=_query_max(frame)
        for s,items in batch.items():
            values[s].append(items);times[s].append(np.full(len(items),timestamp));recent[s].add(timestamp,items)
    static={s:finite_sample_quantile(np.concatenate(v),1-cfg.target_coverage) for s,v in values.items()}
    expanding={s:SortedHistory(np.concatenate(values[s]),np.concatenate(times[s])) for s in values}
    histograms=np.zeros((len(METHODS),3,vocabulary+1),dtype=np.int64)
    strata={};timestamp_rows=[];maximum_nc_difference=0.
    for timestamp,frame in test.groupby('timestamp',sort=True):
        timestamp=int(timestamp);facts=frame[FACT_COLUMNS].to_numpy(dtype=np.int64)
        if len(np.unique(facts,axis=0))!=len(facts): raise ValueError('duplicate observed fact')
        grouping=build_query_grouping(facts)
        raw=_score_all_objects(model,facts,cfg.batch_size)
        scores=raw[grouping.first_indices].astype(float)
        order,ranks=candidate_order(scores);true_ranks=ranks[grouping.inverse,facts[:,2]]
        rel=facts[grouping.first_indices,1]
        bins=np.where(grouping.answer_counts==1,0,np.where(grouping.answer_counts==2,1,np.where(grouping.answer_counts<=5,2,3)))
        groups={'answer_count':(bins,4),'direction':((rel>=relations).astype(int),2),'relation_direction':(rel,2*relations)}
        candidates={s:(scores.max(axis=1,keepdims=True)-scores if s=='margin' else _kgcp_candidate_nonconformity(scores,s,cfg.kgcp_temperature)) for s in SCORES}
        for s in SCORES:
            actual=candidates[s][grouping.inverse,facts[:,2]]
            difference=float(np.max(np.abs(actual-frame[s].to_numpy())))
            maximum_nc_difference=max(maximum_nc_difference,difference)
            if difference>1e-10: raise ValueError(f'candidate replay differs from cached {s}: {difference}')
            score_methods=[f'label_{s}_{h}' for h in HISTORIES]
            thresholds=[static[s],sorted_quantile(expanding[s].values,1-cfg.target_coverage),finite_sample_quantile(recent[s].values_before(timestamp).values,1-cfg.target_coverage)]
            if s=='margin':
                score_methods += [f'query_max_margin_{h}' for h in HISTORIES]
                thresholds += [static['query_max_margin'],sorted_quantile(expanding['query_max_margin'].values,1-cfg.target_coverage),finite_sample_quantile(recent['query_max_margin'].values_before(timestamp).values,1-cfg.target_coverage)]
            sizes=prefix_sizes(candidates[s],order,thresholds)
            for index,method in enumerate(score_methods):
                method_index=METHODS.index(method);base_sizes=sizes[:,index]
                result={'case':case,'seed':seed,'timestamp':timestamp,'method':method,'score':s,'history':method.rsplit('_',1)[-1],'vocabulary_size':vocabulary,'threshold':thresholds[index]}
                for scope_index,mask in enumerate((np.ones(grouping.query_count,dtype=bool),grouping.answer_counts==1,grouping.answer_counts>1)):
                    histograms[method_index,scope_index]+=np.bincount(base_sizes[mask],minlength=vocabulary+1)
                for budget in BUDGETS:
                    capped=base_sizes if budget=='unbounded' else np.minimum(base_sizes,int(budget))
                    sufficient=query_sufficient(capped,true_ranks,grouping,vocabulary)
                    result.update({prefix(budget)+name:value for name,value in summarize_sufficient(sufficient,grouping.answer_counts).items()})
                    if budget=='unbounded':
                        for scope,(codes,length) in groups.items():
                            target=strata.setdefault((method,scope),np.zeros((length,len(SUM_FIELDS))))
                            for column in range(len(SUM_FIELDS)): target[:,column]+=np.bincount(codes,weights=sufficient[:,column],minlength=length)
                timestamp_rows.append(result)
        current={s:frame[s].to_numpy() for s in SCORES};current['query_max_margin']=_query_max(frame)
        # Feedback enters only after all methods predict the complete timestamp.
        for s,items in current.items(): expanding[s].add(items,timestamp);recent[s].add(timestamp,items)
    rows=pd.DataFrame(timestamp_rows);old=pd.read_csv(gate_csv)
    for history,column in [('static','static_coverage'),('rolling','rolling_coverage')]:
        new=rows[rows.method.isin([f'label_{s}_{history}' for s in SCORES])].copy()
        new['coverage']=new.covered_label_count/new.label_count
        merged=new.merge(old[['timestamp','score','query_count',column]],on=['timestamp','score'],suffixes=('','_old'),validate='one_to_one')
        if len(merged)!=len(old) or not (merged.label_count==merged.query_count_old).all() or np.max(np.abs(merged.coverage-merged[column]))>1e-10: raise ValueError('utility label coverage disagrees with P0')
    strata_rows=[]
    for (method,scope),sums in strata.items():
        for index in np.flatnonzero(sums[:,2]>0):
            label=(['1','2','3_5','gt5'][index] if scope=='answer_count' else ['tail','head'][index] if scope=='direction' else f'{index%relations}:{"head" if index>=relations else "tail"}')
            strata_rows.append({'case':case,'seed':seed,'method':method,'scope':scope,'stratum':label,**dict(zip(SUM_FIELDS,sums[index]))})
    provenance={'case':case,'seed':seed,'checkpoint_sha256':checkpoint_hash,'config_sha256':config_hash,'dataset_sha256':dataset['sha256'],'cache_sha256':source_marker['scores_sha256'],'gate_rows_sha256':source_marker['output_sha256'],'candidate_replay_max_difference':maximum_nc_difference,'timestamp_count':int(test.timestamp.nunique()),'row_count':len(rows),'seconds':time.monotonic()-started}
    return rows,pd.DataFrame(strata_rows),histograms,provenance

def seed_summaries(rows,histograms):
    records=[]
    for method_index,method in enumerate(METHODS):
        frame=rows[rows.method==method];case=frame.case.iloc[0];seed=int(frame.seed.iloc[0]);vocabulary=int(frame.vocabulary_size.iloc[0])
        for budget in BUDGETS:
            pref=prefix(budget);record={'case':case,'seed':seed,'method':method,'budget':budget,'vocabulary_size':vocabulary}
            for metric,(num,den) in RATIOS.items():
                denominator=float(frame[pref+den].sum());record[metric]=float(frame[pref+num].sum()/denominator) if denominator>0 else np.nan
            record['mean_set_fraction']=record['mean_set_size']/vocabulary
            for scope_index,scope in enumerate(('overall','single','multi')):
                counts=histograms[method_index,scope_index]
                if budget!='unbounded': counts=clipped_histogram(counts,int(budget))
                expected=frame['query_count' if scope=='overall' else f'{scope}_query_count'].sum()
                if counts.sum()!=expected: raise ValueError('histogram count conservation failed')
                record[f'{scope}_query_count']=int(counts.sum())
                record[f'{scope}_median_set_size']=histogram_quantile(counts,.5) if counts.sum() else np.nan
                record[f'{scope}_p90_set_size']=histogram_quantile(counts,.9) if counts.sum() else np.nan
                if scope=='overall' and not np.isclose(np.dot(np.arange(len(counts)),counts)/counts.sum(),record['mean_set_size'],rtol=0,atol=1e-9): raise ValueError('histogram size sum mismatch')
            records.append(record)
    return pd.DataFrame(records)

def export_bootstrap(rows,out):
    records=[]
    for case_index,(case,frame) in enumerate(rows.groupby('case',sort=True)):
        seeds=sorted(frame.seed.unique());timestamps=sorted(frame.timestamp.unique())
        grid=pd.MultiIndex.from_product([seeds,timestamps],names=['seed','timestamp'])
        metadata=[];numerators=[];denominators=[]
        for method in METHODS:
            selected=frame[frame.method==method].set_index(['seed','timestamp']).reindex(grid)
            if selected.query_count.isna().any(): raise ValueError('missing utility timestamp cell')
            for budget in BUDGETS:
                for metric,(num,den) in RATIOS.items():
                    metadata.append({'method':method,'budget':budget,'metric':metric,'kind':'absolute','reference':''})
                    numerators.append(selected[prefix(budget)+num].to_numpy())
                    denominators.append(selected[prefix(budget)+den].to_numpy())
        index={(r['method'],r['budget'],r['metric']):i for i,r in enumerate(metadata)}
        contrasts=[];contrast_metadata=[]
        for method in [f'label_{s}_rolling' for s in SCORES]+['query_max_margin_rolling']:
            reference=method.replace('_rolling','_static') if method.startswith('label_') else 'label_margin_rolling'
            for budget in BUDGETS:
                for metric in RATIOS:
                    contrasts.append((index[method,budget,metric],index[reference,budget,metric]))
                    contrast_metadata.append({'method':method,'budget':budget,'metric':metric,'kind':'contrast','reference':reference})
        metadata+=contrast_metadata
        nums=np.stack(numerators,axis=-1).reshape(len(seeds),len(timestamps),-1)
        dens=np.stack(denominators,axis=-1).reshape(len(seeds),len(timestamps),-1)
        for block in (3,7,14,21):
            result=bootstrap_ratios(nums,dens,block_length=block,iterations=20000,bootstrap_seed=20260908+1000*case_index+block,contrasts=contrasts)
            for i,meta in enumerate(metadata): records.append({'case':case,'block_length':block,**meta,**{key:float(value[i]) for key,value in result.items()}})
            print(f'[utility bootstrap] {case} block={block}',flush=True)
    result=pd.DataFrame(records);result.to_csv(out/'utility_bootstrap.csv',index=False)
    return result

def verify_directory(root):
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha256_file(root/name)!=expected: raise ValueError(f'checksum failed: {name}')

def condition_job(job):
    case,seed,source_record,gate_root,out,spec_hash=job
    torch.set_num_threads(4)
    key=f'{case}_seed{seed}';base=out/'conditions'/key
    files={'timestamps':base.with_suffix('.parquet'),'strata':base.with_suffix('.strata.parquet'),'histograms':base.with_suffix('.histograms.npz'),'summary':base.with_suffix('.summary.csv')}
    marker_path=base.with_suffix('.complete.json')
    if marker_path.exists():
        record=json.loads(marker_path.read_text())
        if record['specification_sha256']!=spec_hash: raise ValueError('resume source/spec changed')
        for name,path in files.items():
            if sha256_file(path)!=record['outputs'][name]: raise ValueError('resume output changed')
        if sha256_file(gate_root/'scores'/f'{key}.parquet')!=record['cache_sha256']: raise ValueError('resume cache changed')
        checkpoint=Path(source_record['run_root'])/'checkpoints'/f'seed{seed}_delete0p00.ckpt'
        if sha256_file(checkpoint)!=record['checkpoint_sha256']: raise ValueError('resume checkpoint changed')
        print(f'[utility] verified {key}',flush=True)
        return record
    print(f'[utility] starting {key}',flush=True)
    rows,strata,histograms,record=evaluate_condition(case,seed,source_record,gate_root)
    summary=seed_summaries(rows,histograms);base.parent.mkdir(parents=True,exist_ok=True)
    for name,frame in [('timestamps',rows),('strata',strata)]:
        tmp=files[name].with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(files[name])
    tmp=files['histograms'].with_suffix('.tmp')
    with tmp.open('wb') as handle: np.savez_compressed(handle,histograms=histograms,methods=np.asarray(METHODS),scopes=np.asarray(['overall','single','multi']))
    tmp.replace(files['histograms']);summary.to_csv(files['summary'],index=False)
    record.update(status='complete',specification_sha256=spec_hash,outputs={name:sha256_file(path) for name,path in files.items()},torch_threads=4)
    json_write(marker_path,record)
    print(f'[utility] completed {key}: {record["seconds"]:.1f}s',flush=True)
    return record

def main():
    ap=ArgumentParser();ap.add_argument('--matrix-root',type=Path,required=True);ap.add_argument('--gate-root',type=Path,required=True)
    ap.add_argument('--p1-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--analysis-git-commit',required=True)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    if len(a.analysis_git_commit)!=40: raise ValueError('full source commit required')
    if not 1<=a.workers<=4: raise ValueError('workers must be 1 through 4')
    for phase in ('online','feedback'):
        root=a.p1_root/phase;m=json.loads((root/'stage_manifest.json').read_text())
        if m['status']!='complete' or m['condition_count']!=80: raise ValueError('complete online and feedback stages required')
        verify_directory(root)
    gate=json.loads((a.gate_root/'gate_manifest.json').read_text())
    if gate['status']!='complete' or gate['condition_count']!=80: raise ValueError('complete gate required')
    progress=json.loads((a.matrix_root/'matrix_progress.json').read_text())
    paths=[Path(__file__),repo/'scripts/utility_bootstrap.py',repo/'scripts/export_cached_calibration_followups.py',repo/'README.md',*sorted((repo/'src/riskcal_tkg').glob('*.py'))]
    spec={'analysis_git_commit':a.analysis_git_commit,'training_git_commit':progress['git_commit'],'cases':list(EXPECTED_RUNS),'seeds':list(SEEDS),'methods':list(METHODS),'budgets':list(BUDGETS),'candidate_tie_rule':'raw score descending then entity ID ascending','score_precision':'float64 nonconformity from original CPU float32 candidate scores','gate_manifest_sha256':sha256_file(a.gate_root/'gate_manifest.json'),'p1_manifests':{p:sha256_file(a.p1_root/p/'stage_manifest.json') for p in ('online','feedback')},'source_hashes':{str(p.relative_to(repo)):sha256_file(p) for p in paths},'iterations':20000,'block_lengths':[3,7,14,21]}
    spec['runtime']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'device':'cpu','workers':a.workers,'torch_threads_per_worker':4}
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if (out/'specification.json').exists() and json.loads((out/'specification.json').read_text())!=spec: raise ValueError('changed utility specification')
    json_write(out/'specification.json',spec);torch.set_num_threads(4)
    jobs=[]
    for case in EXPECTED_RUNS:
        if progress['runs'][case]['status']!='complete': raise ValueError('incomplete source case')
        for seed in SEEDS:
            jobs.append((case,seed,progress['runs'][case],a.gate_root,out,spec_hash))
    provenance=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(condition_job,job) for job in jobs]
        try:
            for future in as_completed(futures): provenance.append(future.result())
        except BaseException:
            for future in futures: future.cancel()
            raise
    provenance.sort(key=lambda record:(record['case'],record['seed']))
    frames=[];strata_frames=[];summaries=[]
    for record in provenance:
        base=out/'conditions'/f'{record["case"]}_seed{record["seed"]}'
        frames.append(pd.read_parquet(base.with_suffix('.parquet')))
        strata_frames.append(pd.read_parquet(base.with_suffix('.strata.parquet')))
        summaries.append(pd.read_csv(base.with_suffix('.summary.csv')))
    rows=pd.concat(frames,ignore_index=True);strata=pd.concat(strata_frames,ignore_index=True);by_seed=pd.concat(summaries,ignore_index=True)
    rows.to_parquet(out/'utility_by_timestamp.parquet',index=False);strata.to_parquet(out/'utility_strata_sums.parquet',index=False);by_seed.to_csv(out/'utility_by_seed.csv',index=False)
    summary=by_seed.groupby(['case','method','budget'],as_index=False).mean(numeric_only=True).drop(columns='seed')
    summary.to_csv(out/'utility_summary.csv',index=False)
    bootstrap=export_bootstrap(rows,out)
    json_write(out/'utility_manifest.json',{'status':'complete','created_at':datetime.now(UTC).isoformat(),'condition_count':80,'timestamp_row_count':len(rows),'strata_row_count':len(strata),'bootstrap_row_count':len(bootstrap),'specification':spec,'specification_sha256':spec_hash,'provenance':provenance,'quantile_note':'Per-seed pooled unique-query histograms; summary averages per-seed quantiles, not timestamp quantiles.','inference_note':'Pointwise paired circular block intervals. Zero-count timestamps preserved; undefined resamples explicitly counted. No guarantee for budget truncation.'})
    inventory=[f'{sha256_file(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt']
    (out/'SHA256SUMS.txt').write_text('\n'.join(inventory)+'\n');print('UTILITY_COMPLETE',out,flush=True)

if __name__=='__main__': main()
