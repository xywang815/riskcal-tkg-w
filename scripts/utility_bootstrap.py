"""Paired block inference from sufficient sums, preserving zero-count dates."""
import math
import warnings
import numpy as np

def bootstrap_ratios(numerators,denominators,*,block_length,iterations=20000,
                     bootstrap_seed=20260908,contrasts=(),chunk_size=128):
    numerators=np.asarray(numerators,dtype=float);denominators=np.asarray(denominators,dtype=float)
    if numerators.ndim!=3 or numerators.shape!=denominators.shape: raise ValueError('seed x timestamp x metric arrays required')
    if not np.isfinite(numerators).all() or not np.isfinite(denominators).all() or (denominators<0).any(): raise ValueError('invalid sufficient sums')
    seeds,timestamps,metrics=numerators.shape
    with np.errstate(invalid='ignore',divide='ignore'):
        observed=(numerators.sum(axis=1)/denominators.sum(axis=1)).mean(axis=0)
    samples=np.empty((iterations,metrics+len(contrasts)))
    rng=np.random.default_rng(bootstrap_seed);blocks=math.ceil(timestamps/block_length)
    for start in range(0,iterations,chunk_size):
        stop=min(start+chunk_size,iterations);count=stop-start
        seed_draw=rng.integers(0,seeds,size=(count,seeds))
        starts=rng.integers(0,timestamps,size=(count,blocks))
        positions=((starts[:,:,None]+np.arange(block_length))%timestamps).reshape(count,-1)[:,:timestamps]
        seed_counts=np.zeros((count,seeds));time_counts=np.zeros((count,timestamps))
        np.add.at(seed_counts,(np.arange(count)[:,None],seed_draw),1)
        np.add.at(time_counts,(np.arange(count)[:,None],positions),1)
        num=np.einsum('bt,stm->bsm',time_counts,numerators,optimize=True)
        den=np.einsum('bt,stm->bsm',time_counts,denominators,optimize=True)
        with np.errstate(invalid='ignore',divide='ignore'):
            ratios=num/den
        selected=np.where(seed_counts[:,:,None]>0,ratios,0)
        mean=(selected*seed_counts[:,:,None]).sum(axis=1)/seeds
        samples[start:stop,:metrics]=mean
        for j,(left,right) in enumerate(contrasts): samples[start:stop,metrics+j]=mean[:,left]-mean[:,right]
    observed=np.r_[observed,[observed[left]-observed[right] for left,right in contrasts]]
    valid=np.isfinite(samples).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        low=np.nanquantile(samples,.025,axis=0);high=np.nanquantile(samples,.975,axis=0)
    return {'observed':observed,'ci95_low':low,'ci95_high':high,
            'valid_resamples':valid,'undefined_resamples':iterations-valid}
