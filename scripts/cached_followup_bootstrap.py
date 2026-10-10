"""Several paired contrasts from exactly the original circular-block draws."""
import math
import numpy as np

def bootstrap_many(values, weights, *, block_length, iterations=20000,
                   bootstrap_seed=20260908, chunk_size=128):
    values = np.asarray(values,dtype=float)
    weights = np.asarray(weights,dtype=float)
    if values.ndim != 3 or weights.shape != values.shape[:2]:
        raise ValueError('expected seed x timestamp x contrast values and matching weights')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('finite values and positive weights required')
    if block_length < 1 or iterations < 1: raise ValueError('invalid bootstrap sizes')
    seed_count, timestamp_count, contrast_count = values.shape
    observed = ((values*weights[:,:,None]).sum(axis=1)/weights.sum(axis=1)[:,None]).mean(axis=0)
    samples = np.empty((iterations,contrast_count))
    rng = np.random.default_rng(bootstrap_seed)
    blocks = math.ceil(timestamp_count/block_length)
    offsets = np.arange(block_length)
    for start in range(0,iterations,chunk_size):
        stop=min(start+chunk_size,iterations); count=stop-start
        selected_seeds=rng.integers(0,seed_count,size=(count,seed_count))
        starts=rng.integers(0,timestamp_count,size=(count,blocks))
        positions=((starts[:,:,None]+offsets)%timestamp_count).reshape(count,-1)[:,:timestamp_count]
        seed_counts=np.zeros((count,seed_count))
        time_counts=np.zeros((count,timestamp_count))
        np.add.at(seed_counts,(np.arange(count)[:,None],selected_seeds),1)
        np.add.at(time_counts,(np.arange(count)[:,None],positions),1)
        effective=time_counts[:,None,:]*weights[None,:,:]
        effective/=effective.sum(axis=2,keepdims=True)
        effective*=seed_counts[:,:,None]/seed_count
        samples[start:stop]=effective.reshape(count,-1)@values.reshape(-1,contrast_count)
    return {'observed':observed,'ci95_low':np.quantile(samples,.025,axis=0),
            'ci95_high':np.quantile(samples,.975,axis=0),
            'bootstrap_nonpositive_fraction':(np.sum(samples<=0,axis=0)+1)/(iterations+1)}
