import numpy as np
import pandas as pd
import pytest

from riskcal_tkg.calibration import kgcp_nonconformity, margin_nonconformity
from scripts.export_normalized_time_gate import SCORES, true_nonconformity, replay_coverage

def test_true_scores_match_original_candidate_definitions():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(10,20)).astype(np.float32)
    scores[0] = 3.0
    labels = np.arange(10)
    observed = true_nonconformity(scores,labels)
    np.testing.assert_allclose(observed['margin'],margin_nonconformity(scores,labels),atol=0,rtol=0)
    for method in ('negscore','minmax','softmax'):
        np.testing.assert_allclose(observed[method],kgcp_nonconformity(scores,labels,method),atol=1e-15,rtol=0)

def test_minmax_is_invariant_to_per_query_positive_affine_transforms():
    scores = np.array([[3.,1.,2.],[-1.,7.,0.]])
    changed = scores*np.array([[2.],[0.2]])+np.array([[100.],[-30.]])
    np.testing.assert_allclose(true_nonconformity(scores,[0,1])['minmax'],true_nonconformity(changed,[0,1])['minmax'])

def test_replay_never_uses_current_timestamp_labels():
    calibration = pd.DataFrame({'timestamp':[1]*10,**{s:[0.]*10 for s in SCORES}})
    test = pd.DataFrame({'timestamp':[2]*10+[3],**{s:[100.]*11 for s in SCORES}})
    rows = replay_coverage(calibration,test,window=10)
    assert (rows[rows.timestamp == 2].rolling_coverage == 0).all()
    assert (rows[rows.timestamp == 3].rolling_coverage == 1).all()
    assert (rows.static_coverage == 0).all()
    assert (rows.calibration_max_timestamp < rows.timestamp).all()
    with pytest.raises(ValueError,match='overlap'):
        replay_coverage(calibration,test.assign(timestamp=1))
