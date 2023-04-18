import os

import numpy as np
import pandas as pd

import anndata


"""
Create a dataset with binary HMM
"""


def create_adata(x,metdata,logdir,filename='anndata.h5ad',save=True):
    """"""
    adata = anndata.AnnData(x)
    adata.obs = metadata
    adata.var['Feature'] = 'X'
    if save:
        adata.write(os.path.join(logdir,filename))
    return adata

def generate_hmm_sequence(x,n,prob1=0.4,prob2=0.1):
    """
    Create a dataset with look back dependency on top of binary HMM
        prob1 = transition probability from 1 to 0
        prob2 = transition probability from 0 to 1
    """
    out = []
    for _ in range(n):
        if x == 1:
            weight = (1-prob1,prob1)
        else:
            weight = (prob2,1-prob2)
        x = np.random.choice([1,0],1,True,weight)
        out.append(x)
    return np.array(out)

def generate_hmm_sequence_w_longterm_dependency(x,n,l,prob1=0.4,prob2=0.1,prob3=0.7):
    """
    Input:
        x: array
        n: int number of samples
        l: int length of look back
        prob3: probability of transition from 0 to 1 dependent on long-term dependency
    """
    for _ in range(n):
        if len(x)-1 > l:
            if x[-l] == 1 and x[-1] == 0:
                xnew = generate_hmm_sequence(x[-l],1,prob3)
            else:
                xnew = generate_hmm_sequence(x[-1],1,prob1,prob2)
        else:
            xnew = generate_hmm_sequence(x[-1],1,prob1,prob2)
        x = np.append(x,xnew)
    return x[-n:]

def generate_sequence(x, n_seqs, seq_len, mode = 'bernoulli', **kwargs):
    """
    Input
        x: starting sequence
    """
    if isinstance(n_seqs,int):
        n_seqs = [n_seqs]*3
    X        = []
    metadata = []
    datatypes = ['Train','Test','Eval'][:len(n_seqs)]
    for datatype,n_seq in zip(datatypes,n_seqs):
        for i_sample in range(n_seq):
            if mode == 'bernoulli':
                xs = np.random.choice([0,1],seq_len,True).reshape(-1,1)
            elif mode == 'hmm':
                xs = generate_hmm_sequence(x,seq_len,**kwargs)
            elif mode == 'hmm-long':
                xs = generate_hmm_sequence_w_longterm_dependency(x,seq_len,**kwargs).reshape(-1,1)
            else:
                raise ValueError('Input valid mode: {}'.format(mode))
            X.append(xs)
            df_meta = pd.DataFrame({'Mode'     : mode,
                                    'Train'    : datatype,
                                    'Sample'   : 'Sample-{}'.format(i_sample),
                                    'Time'     : range(len(xs))})
            metadata.append(df_meta)
    X = np.concatenate(X,axis=0)
    metadata = pd.concat(metadata).reset_index(drop=True)
    return X,metadata

logdir = 'data'
os.makedirs(logdir,exist_ok=True)

####
x = [0]
n_seqs = [1,1,1] # train,test,eval
seq_len = 10000

"""
Create a dataset with binary activity with 0.5 chance probability
"""

X, metadata = generate_sequence(x, n_seqs, seq_len, mode='bernoulli')
logdir_d = os.path.join(logdir,'simulation1')
os.makedirs(logdir_d,exist_ok=True)
adata = create_adata(X, metadata,logdir=logdir_d,filename='simulation.h5ad')

X, metadata = generate_sequence(x, n_seqs, seq_len, mode='bernoulli',
                                prob1=0.4,prob2=0.1)
logdir_d = os.path.join(logdir,'simulation2')
os.makedirs(logdir_d,exist_ok=True)
adata = create_adata(X, metadata,logdir=logdir_d,filename='simulation.h5ad')

X, metadata = generate_sequence(x, n_seqs, seq_len, mode='hmm',
                                prob1=0.05,prob2=0.05)
logdir_d = os.path.join(logdir,'simulation2_2')
os.makedirs(logdir_d,exist_ok=True)
adata = create_adata(X, metadata,logdir=logdir_d,filename='simulation.h5ad')

X, metadata = generate_sequence(x, n_seqs, seq_len, mode='hmm-long',
                                prob1=0.05,prob2=0.05,prob3=0.7,l=50)
logdir_d = os.path.join(logdir,'simulation3')
os.makedirs(logdir_d,exist_ok=True)
adata = create_adata(X, metadata,logdir=logdir_d,filename='simulation.h5ad')
