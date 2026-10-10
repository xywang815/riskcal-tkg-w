import math
import numpy as np
import pytest
from riskcal_tkg.calibration import _kgcp_candidate_nonconformity
from riskcal_tkg.followup import build_query_grouping,summarize_query_mask
from riskcal_tkg.utility_frontier import candidate_order,prefix_sizes,query_sufficient,summarize_sufficient,histogram_quantile,clipped_histogram
from scripts.utility_bootstrap import bootstrap_ratios
from scripts.cached_followup_bootstrap import bootstrap_many

@pytest.mark.parametrize('score',['margin','negscore','minmax','softmax'])
def test_prefix_and_budget_metrics_equal_explicit_masks_with_ties(score):
    facts=np.array([[0,0,1,7],[0,0,3,7],[2,1,0,7],[3,1,2,7]])
    group=build_query_grouping(facts)
    scores=np.array([[2.,2.,0.,1.,-1.],[0.,0.,0.,0.,0.],[1.,3.,3.,-2.,0.]])
    order,ranks=candidate_order(scores)
    nc=scores.max(axis=1,keepdims=True)-scores if score=='margin' else _kgcp_candidate_nonconformity(scores,score)
    thresholds=np.array([-np.inf,.5,np.inf])
    sizes=prefix_sizes(nc,order,thresholds)
    true_ranks=ranks[group.inverse,facts[:,2]]
    for j,threshold in enumerate(thresholds):
        for budget in (0,1,2,5):
            clipped=np.minimum(sizes[:,j],budget)
            summary=summarize_sufficient(query_sufficient(clipped,true_ranks,group,5),group.answer_counts)
            mask=(nc<=threshold)&(ranks<=budget)
            expected=summarize_query_mask(mask,facts,group)
            assert summary['covered_label_count']/summary['label_count']==expected['label_coverage']
            assert summary['full_query_count']/summary['query_count']==expected['full_set_coverage']
            assert summary['set_size_sum']/summary['query_count']==expected['mean_set_size']
    assert order[1].tolist()==[0,1,2,3,4]

def test_histogram_quantiles_are_pooled_and_budget_clipping_is_exact():
    values=np.array([0,0,2,6,7,7,10]);hist=np.bincount(values,minlength=11)
    for budget in (1,5,10):
        clipped=clipped_histogram(hist,budget)
        assert clipped.sum()==len(values)
        for q in (.1,.5,.9,1):
            assert histogram_quantile(clipped,q)==pytest.approx(np.quantile(np.minimum(values,budget),q))
    assert histogram_quantile(np.array([1,0,1]),.5)==1
    assert histogram_quantile(clipped_histogram(np.array([1,0,1]),1),.5)==.5

def test_ratio_bootstrap_matches_original_positive_weight_estimator():
    rng=np.random.default_rng(4);values=rng.normal(size=(3,11,2));weights=rng.integers(1,20,size=(3,11))
    original=bootstrap_many(values,weights,block_length=3,iterations=257,bootstrap_seed=7)
    actual=bootstrap_ratios(values*weights[:,:,None],np.broadcast_to(weights[:,:,None],values.shape),block_length=3,iterations=257,bootstrap_seed=7)
    for name in ('observed','ci95_low','ci95_high'):
        np.testing.assert_allclose(actual[name],original[name],atol=1e-12)

def test_zero_count_dates_remain_in_block_draws_and_undefined_draws_are_counted():
    num=np.array([[[0.],[2.]]]);den=np.array([[[0.],[4.]]])
    actual=bootstrap_ratios(num,den,block_length=1,iterations=128,bootstrap_seed=5)
    rng=np.random.default_rng(5);rng.integers(0,1,size=(128,1))
    positions=rng.integers(0,2,size=(128,2))
    undefined=np.sum((positions==0).all(axis=1))
    assert actual['undefined_resamples'][0]==undefined
    assert actual['observed'][0]==.5
    assert actual['ci95_low'][0]==.5 and actual['ci95_high'][0]==.5

def test_full_export_preserves_metric_order_and_paired_contrasts(tmp_path,monkeypatch):
    import pandas as pd
    from scripts import export_candidate_utility as exporter
    records=[]
    for seed in (17,29):
        for timestamp in (3,4,5):
            for method in exporter.METHODS:
                record={'case':'synthetic','seed':seed,'timestamp':timestamp,'method':method}
                for budget in exporter.BUDGETS:
                    pref=exporter.prefix(budget)
                    counts={'label_count':10,'covered_label_count':8,'query_count':5,'full_query_count':3,
                            'single_query_count':2,'single_full_query_count':1,'multi_query_count':3,'multi_full_query_count':2,
                            'set_size_sum':50,'single_set_size_sum':20,'multi_set_size_sum':30,
                            'partial_recall_sum':4,'full_vocabulary_query_count':0}
                    record.update({pref+name:value*(1+seed%3) for name,value in counts.items()})
                records.append(record)
    original=exporter.bootstrap_ratios
    def small_bootstrap(*args,**kwargs):
        assert kwargs['iterations']==20000
        kwargs['iterations']=257
        return original(*args,**kwargs)
    monkeypatch.setattr(exporter,'bootstrap_ratios',small_bootstrap)
    result=exporter.export_bootstrap(pd.DataFrame(records),tmp_path)
    assert len(result)==4*(15+5)*6*9
    expected={'label_coverage':.8,'full_set_coverage':.6,'single_full_set_coverage':.5,
              'multi_full_set_coverage':2/3,'mean_set_size':10,'single_mean_set_size':10,
              'multi_mean_set_size':10,'partial_answer_recall':.8,'full_vocabulary_rate':0}
    for metric,value in expected.items():
        selected=result[(result.kind=='absolute')&(result.metric==metric)]
        np.testing.assert_allclose(selected[['observed','ci95_low','ci95_high']],value,atol=1e-12)
    np.testing.assert_allclose(result[result.kind=='contrast'][['observed','ci95_low','ci95_high']],0,atol=1e-12)
    assert (result.valid_resamples==257).all() and (result.undefined_resamples==0).all()
