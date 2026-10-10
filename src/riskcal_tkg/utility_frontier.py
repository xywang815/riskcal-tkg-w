"""Exact unique-query set statistics from a deterministic candidate ordering."""
import numpy as np

SUM_FIELDS=('label_count','covered_label_count','query_count','full_query_count',
            'partial_recall_sum','set_size_sum','full_vocabulary_query_count')

def candidate_order(scores):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=2 or not np.isfinite(scores).all(): raise ValueError('finite candidate matrix required')
    order=np.argsort(-scores,axis=1,kind='stable')
    ranks=np.empty_like(order)
    ranks[np.arange(len(scores))[:,None],order]=np.arange(scores.shape[1])+1
    return order,ranks

def prefix_sizes(candidate_nc,order,thresholds):
    ordered=np.take_along_axis(candidate_nc,order,axis=1)
    if (np.diff(ordered,axis=1)<0).any(): raise ValueError('nonconformity must be monotone in candidate order')
    return np.stack([np.searchsorted(row,thresholds,side='right') for row in ordered])

def query_sufficient(sizes,true_ranks,grouping,vocabulary_size):
    sizes=np.asarray(sizes,dtype=int)
    if sizes.shape!=(grouping.query_count,) or ((sizes<0)|(sizes>vocabulary_size)).any(): raise ValueError('invalid set sizes')
    covered=np.asarray(true_ranks)<=sizes[grouping.inverse]
    counts=np.bincount(grouping.inverse,weights=covered,minlength=grouping.query_count)
    return np.column_stack([grouping.answer_counts,counts,np.ones(len(sizes)),
        counts==grouping.answer_counts,counts/grouping.answer_counts,sizes,sizes==vocabulary_size])

def summarize_sufficient(values,answer_counts):
    result=dict(zip(SUM_FIELDS,values.sum(axis=0)))
    for name,mask in [('single',answer_counts==1),('multi',answer_counts>1)]:
        subtotal=values[mask].sum(axis=0)
        result[f'{name}_query_count']=float(subtotal[2])
        result[f'{name}_full_query_count']=float(subtotal[3])
        result[f'{name}_set_size_sum']=float(subtotal[5])
    return result

def histogram_quantile(histogram,probability):
    counts=np.asarray(histogram,dtype=np.int64)
    if (counts<0).any() or counts.sum()==0 or not 0<=probability<=1: raise ValueError('invalid histogram')
    position=(int(counts.sum())-1)*probability
    lower=int(np.floor(position));upper=int(np.ceil(position))
    cumulative=np.cumsum(counts)
    left=int(np.searchsorted(cumulative,lower+1,side='left'))
    right=int(np.searchsorted(cumulative,upper+1,side='left'))
    return float(left+(position-lower)*(right-left))

def clipped_histogram(histogram,budget):
    counts=np.asarray(histogram,dtype=np.int64)
    if budget<0: raise ValueError('negative budget')
    result=np.zeros_like(counts)
    cap=min(int(budget),len(counts)-1)
    result[:cap]=counts[:cap];result[cap]=counts[cap:].sum()
    return result
