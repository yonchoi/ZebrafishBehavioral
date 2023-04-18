import numpy as np
import pandas as pd

import anndata

def generate_samples(X, obs, n_sample=100, seq_len=128,
                     window_len=1, window_overlap=None,
                     mode='sample',stratify=['Location'], subset=None):
    """ Generate sequential samples from adata """

    if window_overlap is None:
        window_overlap = seq_len-1

    if window_len != 1:
        X, obs, _ = generate_samples(X, obs,
                                     n_sample=None,
                                     seq_len=window_len,
                                     window_overlap=window_overlap,
                                     mode='sequence',
                                     stratify=stratify,
                                     subset=subset)
        X, obs, attn_mask = generate_samples(X, obs,
                                             n_sample=n_sample,
                                             seq_len=seq_len,
                                             mode=mode,
                                             stratify=stratify,
                                             subset=subset)
        sample   = X.reshape(*X.shape[:2],-1)
        metadata = obs
    else:
        if subset is not None:
            for k,v in subset.items():
                idx_  = np.isin(obs[k],v)
                X     = X[idx_]
                obs   = obs[idx_]

        metadata = obs.drop_duplicates(subset=stratify)[stratify]

        ## Iterate through unique combination of categories and sample
        metadata_list = []
        sample_list   = []

        for _,comb in metadata.iterrows():

            # Iterate through and subset per category
            X_subset = None
            for col in metadata.columns:
                if X_subset is None:
                    # adata_subset = adata[adata.obs[col] == comb[col]]
                    idx_ = obs[col] == comb[col]
                    X_subset = X[idx_]
                    obs_subset = obs[idx_]
                else:
                    idx_ = obs_subset[col] == comb[col]
                    X_subset   = X_subset[idx_]
                    obs_subset = obs_subset[idx_]

            # Generate samples
            idx_valid = np.arange(len(X_subset)-seq_len+1)
            idx_valid = idx_valid[::(seq_len-window_overlap)]
            if mode == 'sample':
                if n_sample is None:
                    raise ValueError("n_sample cannot be None in sample mode")
                idx = np.random.choice(idx_valid, n_sample, replace = True)
            elif mode == 'sequence':
                if n_sample is None:
                    # if n_sample is None, choose every valid idx
                    n_sample = len(idx_valid)
                idx = idx_valid[np.linspace(0,len(idx_valid)-1,n_sample).astype('int')]
            else:
                raise ValueError('Input valid mode ([sample,sequence])')

            sampled = np.array([X_subset[ii:ii+seq_len] for ii in idx])

            ## Add result to list
            sample_list.append(sampled)
            metadata_list.append(obs_subset.iloc[idx])

        sample   = np.concatenate(sample_list)
        metadata = pd.concat(metadata_list)

        # mask  = [np.concatenate([np.ones(5),np.zeros(seq_len-5)])] # test mask
        attn_mask  = np.array([np.ones(seq_len)]*len(sample))

    return sample, metadata, attn_mask
