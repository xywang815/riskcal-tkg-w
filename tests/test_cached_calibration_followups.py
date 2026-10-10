import numpy as np
import pandas as pd
import pytest

from riskcal_tkg.calibration import finite_sample_quantile,weighted_quantile
from scripts.cached_followup_bootstrap import bootstrap_many
from scripts.export_time_encoder_audit import bootstrap_statistic
from scripts.export_normalized_time_gate import replay_coverage
from scripts.export_cached_calibration_followups import SCORES,SortedHistory,sorted_quantile,online_rows,feedback_rows,verify_against_gate

def test_vectorized_bootstrap_matches_original_draws_with_unequal_seed_weights():
    rng=np.random.default_rng(2)
    values=rng.normal(size=(3,13,2));weights=rng.integers(1,100,size=(3,13))
    result=bootstrap_many(values,weights,block_length=4,iterations=257,bootstrap_seed=9)
    for column in range(2):
        frame=pd.DataFrame([{'seed':s,'timestamp':t,'query_count':weights[s,t],'effect':values[s,t,column]} for s in range(3) for t in range(13)])
        expected=bootstrap_statistic(frame,'effect',block_length=4,iterations=257,bootstrap_seed=9)
        for name in ('observed','ci95_low','ci95_high'):
            assert result[name][column]==pytest.approx(expected[name],abs=1e-12)

def test_sorted_history_preserves_weighted_and_finite_quantiles_with_ties():
    values=np.array([1.,3.,1.,2.]);times=np.array([1,1,2,2])
    hist=SortedHistory(values,times);hist.add([0.,3.,3.],3)
    values=np.r_[values,[0.,3.,3.]];times=np.r_[times,[3,3,3]]
    for alpha in (.1,.3,.95):
        assert sorted_quantile(hist.values,alpha)==finite_sample_quantile(values,alpha)
    for half in (7.,14.,np.inf):
        weights=np.ones(len(values)) if np.isinf(half) else np.exp2(-(4-times)/half)
        assert hist.weighted_quantile(4,half)==weighted_quantile(values,weights,.9)

def _example():
    rng=np.random.default_rng(7)
    cal=pd.DataFrame({'timestamp':np.repeat(np.arange(1,4),12),**{s:rng.normal(size=36) for s in SCORES}})
    test=pd.DataFrame({'timestamp':np.repeat(np.arange(4,10),17),**{s:rng.normal(size=102) for s in SCORES}})
    return cal,test

def test_online_and_feedback_reference_reproduce_gate_for_every_score():
    cal,test=_example();gate=replay_coverage(cal,test)
    verify_against_gate(online_rows(cal,test),gate,'online')
    verify_against_gate(feedback_rows(cal,test,17),gate,'feedback')

def test_delayed_feedback_is_released_after_whole_batch():
    cal=pd.DataFrame({'timestamp':[1]*1000,**{s:[0.]*1000 for s in SCORES}})
    test=pd.DataFrame({'timestamp':[2]*1000+[3]*1000+[4]*1000,**{s:[10.]*3000 for s in SCORES}})
    rows=feedback_rows(cal,test,17)
    immediate=rows[(rows.score=='margin')&(rows.method=='rolling_delay0_retain1')]
    delayed=rows[(rows.score=='margin')&(rows.method=='rolling_delay1_retain1')]
    assert immediate.coverage.tolist()==[0.,1.,1.]
    assert delayed.coverage.tolist()==[0.,0.,1.]

def test_feedback_masks_are_shared_across_identical_score_definitions():
    cal,test=_example()
    for s in SCORES: cal[s]=cal.margin;test[s]=test.margin
    rows=feedback_rows(cal,test,29)
    assert rows.groupby(['timestamp','method']).coverage.nunique().max()==1
