## System
import os

## Math
import pandas as pd
import numpy as np

import anndata

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

## Custom functions
from Preprocessing.preprocessing import feature_calculation as fc

from Preprocessing.preprocessing.preprocessing import plot_3d
from Preprocessing.preprocessing.preprocessing import remove_zeros
from Preprocessing.preprocessing.preprocessing import interpolate
from Preprocessing.preprocessing.preprocessing import center

from Preprocessing.preprocessing.spectral_analysis import apply_fft
from Preprocessing.preprocessing.spectral_analysis import apply_cwt

trial_num = 11

## datadir
logdir = 'trials/{}'.format(trial_num)

## Load genotype information
df_genotype = pd.read_csv(os.path.join(logdir,'Trial{}_Genotypes.txt'.format(trial_num)),sep='\t')


## Convert location (loc01,loc02,...) label to well label (A1,A2,...)
wells = ["{}{}".format(a,n+1) for a in ['A','B','C','D','E','F','G','H'] for n in range(12)]
well2loc = {w:'Loc{:02d}'.format(i+1) for i,w in enumerate(wells)}
loc2well = {loc:well for well,loc in well2loc.items()}
well2genotype = {well: genotype for well,genotype in zip(df_genotype.Well,df_genotype.Genotype)}

df_list = []
for ii in range(1,10):
    ## Load locomotion tracking info
    df = pd.read_csv(os.path.join(logdir,'output{}.txt'.format(ii)),sep='\t',header=None)
    df.columns = ['abstime','type','Location','X','Y']
    ## Filter out the locations/wells without genotype annotation
    df['Well'] = df.Location.map(loc2well)
    df['Genotype'] = df.Well.map(well2genotype)
    df = df[np.isin(df.Well,df_genotype.Well)]
    df_list.append(df)

df = pd.concat(df_list)
df = df.sort_values(['Location','abstime'])
## Use groupby to apply filtering process per location
# groupby = df.groupby('Location').groups
# groupby = {loc: df.loc[idx] for loc,idx in groupby.items()}
groupby = {loc: df[df.Location==loc] for loc in df.Location.unique()}
groupby = {loc: remove_zeros(df_) for loc, df_ in groupby.items()}
groupby = {loc: interpolate(df_,'linear') for loc, df_ in groupby.items()}
groupby = {loc: center(df_,) for loc, df_ in groupby.items()}
groupby = {loc: df_.round(1) for loc, df_ in groupby.items()}


## Match the start and end time for all locations
start_time = max([min(df.abstime) for df in groupby.values()])
end_time   = min([max(df.abstime) for df in groupby.values()])
groupby = {loc: df_[(df_.abstime >= start_time) & (df_.abstime <= end_time)] for loc, df_ in groupby.items()}

#### ===========================================================================
#### Initial plots
#### ===========================================================================

df_loc = df.drop_duplicates(subset=['Location'])
df_loc = df_loc.groupby('Genotype').first().sort_values('Genotype')
subset_loc = df_loc.Location.to_list()

#### 2D locomotion plot

fig,plts = plt.subplots(8,12,figsize=(36,24))

for ii in np.arange(96)+1:
    name = 'Loc{:02d}'.format(ii)
    nrow = int((ii-1)/12)
    ncol = (ii-1)%12
    ax = plts[nrow][ncol]
    if name in groupby.keys():
        df_ = groupby[name]
        _ = ax.set_title(name)
        ax.plot(df_.X,df_.Y,linewidth=0.5)

plt.savefig(os.path.join(logdir,'locomotion.pdf'))
plt.close()

#### 3D plot
for loc in subset_loc:
    df_ = groupby[loc]
    #
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    _=plot_3d(ax,df_.X, df_.Y, df_.abstime)
    plt.savefig('3D-{}.pdf'.format(loc))
    plt.close()
    # break it into parts
    num_parts = 8
    fig,axes = plt.subplots(num_parts, figsize=(10,10*num_parts),
                            subplot_kw=dict(projection='3d'))
    for ii,ax in enumerate(axes.flat):
        # ax = plt.axes(ax,projection="3d")
        _=plot_3d(ax,df_.X, df_.Y, df_.abstime)
        z=df_.abstime
        zmax = (z.max()-z.min()) * ((ii+1)/num_parts) + z.min()
        zmin = (z.max()-z.min()) * ((ii)/num_parts) + z.min()
        ax.set_zlim(zmin, zmax)
    plt.savefig(os.path.join(logdir,'3D-{}-parts.pdf'.format(loc)))
    plt.close()

#### ===========================================================================
#### Calculate features
#### ===========================================================================

#### Distance
# features = ['Distance from center','Distance','Linear speed', 'Linear acceleration',
#             'Angle', 'Angular speed', 'Angular acceleration']
features = ['Distance from center','Distance','Linear speed', 'Linear acceleration',
            'Angle']

logdir_plots = os.path.join(logdir,'Feature plots')
os.makedirs(logdir_plots,exist_ok=True)
palette = {'WT':'blue',
           'HET':'green',
           'HOMO':'red'}

write_adata = False

for T in [1]:
    for K in [1]:
        for L in [10,25,50]:
            for skip in [1,10,20,60]:
    # for T in [1,25,50]:
    #     for K in [1,25,50]:
            logdir_TK = os.path.join(logdir_plots,"T-{}_K-{}_L-{}_s-{}".format(T,K,L,skip))
            os.makedirs(logdir_TK,exist_ok=True)
            df_list    = []
            adata_list = []
            for loc,df_ in groupby.items():
                t = df_.abstime.to_numpy()
                X = df_.X
                Y = df_.Y
                dist,t_  = fc.calculate_distance(t,X,Y,T,K,L,skip=skip)
                df_list.append(pd.DataFrame({'Value':dist, 'abstime': t_,
                                             'loc': loc, 'Feature': 'Distance'}))
                speed,t_ = fc.calculate_linear_speed(t,X,Y,T,K,L,skip=skip)
                df_list.append(pd.DataFrame({'Value':speed, 'abstime': t_,
                                             'loc': loc, 'Feature': 'Linear speed'}))
                accel,t_ = fc.calculate_acceleration(t,X,Y,T,K,L,skip=skip)
                df_list.append(pd.DataFrame({'Value':accel, 'abstime': t_,
                                             'loc':loc, 'Feature': 'Linear acceleration'}))
                cutoff=0.1
                dist_,_  = fc.calculate_distance(t,X,Y,T,K=1,L=1)
                angle,t_,_ = fc.calculate_angle(t,X,Y,T=T,K=K,L=L,d=dist_,cutoff=cutoff,skip=skip)
                df_list.append(pd.DataFrame({'Value':angle, 'abstime': t_,
                                             'loc': loc, 'Feature': 'Angle'}))
                angle_speed,t_ = fc.calculate_angular_speed(t,X,Y,d=dist_,cutoff=cutoff,T=T,K=K,L=L,skip=skip)
                df_list.append(pd.DataFrame({'Value':angle_speed, 'abstime': t_,
                                             'loc': loc, 'Feature': 'Angular speed'}))
                angle_accel,t_ = fc.calculate_angular_acceleration(t,X,Y,T=T,K=K,L=L,d=dist_,cutoff=cutoff,skip=skip)
                df_list.append(pd.DataFrame({'Value':angle_accel, 'abstime': t_,
                                             'loc': loc, 'Feature': 'Angular acceleration'}))
                dist_c,t_ = fc.calculate_distance_from_center(X,Y,t,L=L,skip=skip)
                df_list.append(pd.DataFrame({'Value':dist_c, 'abstime': t_,
                                             'loc': loc, 'Feature': 'Distance from center'}))
                #### STFT
                if loc in subset_loc:
                    dist_,_  = fc.calculate_distance(t,X,Y,T,K,L=1,skip=1)
                    frame_length=200
                    frame_step=50
                    spectrogram = tf.signal.stft(tf.cast(dist,tf.float32),
                                                 frame_length=frame_length,
                                                 frame_step=frame_step) # (Time point,Frequency)
                    spectrogram = tf.abs(spectrogram).numpy()
                    # t_ = t[int(frame_length-1):-int(frame_length-1)][::frame_step].to_numpy()
                    t_spec = t[np.linspace(0,len(t)-1,spectrogram.shape[0]).astype('int')]
                    def plot_spectrogram(spectrogram, t, ax):
                        # Convert to frequencies to log scale and transpose so that the time is
                        # represented in the x-axis (columns).
                        log_spec = np.log(spectrogram.T)
                        height = log_spec.shape[0]
                        width = log_spec.shape[1]
                        X = t
                        Y = range(height)
                        ax.pcolormesh(X, Y, log_spec)
                    fig, axes = plt.subplots(2, figsize=(12, 8))
                    timescale = np.arange(dist.shape[0])
                    axes[0].plot(timescale, dist)
                    axes[0].set_title('Distance')
                    plot_spectrogram(spectrogram, t_spec, axes[1])
                    axes[1].set_title('Spectrogram')
                    plt.savefig(os.path.join(logdir_TK,'Spectrogram_dist-{}.svg'.format(loc)))
                    plt.close()
                #### Matrix format
                if write_adata:
                    mat = np.array([dist,speed,accel,angle,angle_speed,angle_accel]).transpose()
                    adata = anndata.AnnData(mat)
                    adata.obs['Location'] = loc
                    adata.obs['Time']     = t_
                    adata_list.append(adata)
            ####
            if write_adata:
                adata = anndata.AnnData.concatenate(*adata_list)
                adata.write(os.path.join(logdir_TK,'feature_matrix.h5ad'))
            ####
            df_feature = pd.concat(df_list)
            df_feature['Well'] = df_feature['loc'].map(loc2well)
            df_feature['Genotype'] = df_feature['Well'].map(well2genotype)
            #### Overlay features on 3D plot
            # for loc in subset_loc:
            #     for feature in features:
            #         df_ = groupby[loc] # X,Y
            #         df_feature_ = df_feature[(df_feature.Feature == feature) & (df_feature['loc'] == loc)]
            #         df_feature_.index = df_feature_.abstime
            #         df_feature_ = df_feature_.reindex(df_.abstime)
            #         series_feature = df_feature_.Value.fillna(method='ffill').fillna(method='bfill')
            #         df_[feature] = series_feature.to_numpy()
            #         #
            #         fig = plt.figure()
            #         ax = plt.axes(projection="3d")
            #         _=plot_3d(ax,df_.X, df_.Y, df_.abstime,
            #                   hue=np.abs(df_[feature]),
            #                   cmap=sns.cubehelix_palette(as_cmap=True))
            #         plt.savefig(os.path.join(logdir_TK,'3D-{}-{}.pdf'.format(loc,feature)))
            #         plt.close()
            #### Plot Time domain features with colors
            ## Plot for subset features
            df_subset = df_feature[np.isin(df_feature['loc'],subset_loc)]
            locs     = df_subset['loc'].unique()
            features = df_subset['Feature'].unique()
            ## Colored feature plot
            # fig,axes=plt.subplots(len(features),len(locs),
            #                       figsize=(5*len(features),5*len(locs)))
            # for feature,axesr in zip(features,axes):
            #     for loc,ax in zip(locs,axesr):
            #         df_ = groupby[loc] # X,Y
            #         df_feature_ = df_feature[(df_feature.Feature == feature) & (df_feature['loc'] == loc)]
            #         df_feature_.index=df_feature_.abstime
            #         df_feature_ = df_feature_.reindex(df_.abstime)
            #         series_feature = df_feature_.Value.fillna(method='ffill').fillna(method='bfill')
            #         _ = plot_3d(ax,df_feature_.abstime.index,series_feature,hue=df_feature_.abstime.index)
            # plt.savefig(os.path.join(logdir_TK,'TimeDomain-all_subset-colored.svg'))
            # plt.close()
            ## Plot for all the features
            g = sns.FacetGrid(df_subset, col="loc", row='Feature', hue='Genotype',
                              palette=palette, sharey=False, sharex=True)
            g.map(sns.lineplot, "abstime","Value", palette=palette)
            plt.savefig(os.path.join(logdir_TK,'TimeDomain-all_subset.svg'.format(feature)))
            plt.close()
            # for feature in features:
            #     df_ = df_feature[df_feature.Feature == feature]
                #### Cummulative information
                # g = sns.ecdfplot(data=df_, x='Value',hue='Genotype',palette=palette)
                # plt.savefig(os.path.join(logdir_TK,'CDF-{}.svg'.format(feature)))
                # plt.close()
                # g = sns.displot(df_, x='Value', hue='Genotype', palette=palette, kind='kde')
                # plt.savefig(os.path.join(logdir_TK,'PDF-{}.svg'.format(feature)))
                # plt.close()
                # g = sns.histplot(df_, x='Value', hue='Genotype', bins = 20,
                #                  palette=palette, stat='probability')
                # plt.savefig(os.path.join(logdir_TK,'HISTOGRAM-{}.svg'.format(feature)))
                # plt.close()
                #### Time domain----------------------------------------------------
                # groupby_ = df_.groupby('loc')
                # groupby_ = {g: groupby_.get_group(g) for g in groupby_.groups}
                # ## Plot line plot
                # g = sns.FacetGrid(df_, col="loc", col_wrap=6, hue='Genotype', palette=palette)
                # g.map(sns.lineplot, "abstime","Value", palette=palette)
                # plt.savefig(os.path.join(logdir_TK,'TimeDomain-{}.svg'.format(feature)))
                # plt.close()
                # ## Plot line plot for subset
                # df_subset = df_[np.isin(df_['loc'],['Loc04','Loc13','Loc15'])]
                # g = sns.FacetGrid(df_subset, col="loc", col_wrap=6, hue='Genotype', palette=palette)
                # g.map(sns.lineplot, "abstime","Value", palette=palette)
                # plt.savefig(os.path.join(logdir_TK,'TimeDomain-{}_subset.svg'.format(feature)))
                # plt.close()
                #### Frequency domain ----------------------------------------------
                # df_freq = pd.concat([apply_fft(df_g).assign(loc=g) for g,df_g in groupby_.items()])
                # df_freq['Feature'] = feature
                # df_freq['Well'] = df_freq['loc'].map(loc2well)
                # df_freq['Genotype'] = df_freq.Well.map(well2genotype)
                # ##
                # g = sns.FacetGrid(df_freq, col="loc", col_wrap=6, hue='Genotype', palette=palette)
                # g.map(sns.lineplot, "Freq","Sp", palette=palette)
                # plt.savefig(os.path.join(logdir_TK,'FreqDomain-{}.svg'.format(feature)))
                # plt.close()
                # ##
                # g = sns.FacetGrid(df_freq, col="Genotype", hue='Genotype', palette=palette)
                # g.map(sns.lineplot, "Freq","Sp", palette=palette)
                # plt.savefig(os.path.join(logdir_TK,'FreqDomain2-{}.svg'.format(feature)))
                # plt.close()
                # #### Wavelet
                # scales = np.arange(1,1280,10)
                # wavelet='gaus8'
                # df_list = {g: apply_cwt(scales,wavelet,df=df_g) for g,df_g in groupby_.items()}
                # cmap = plt.cm.seismic
                # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
                # contourlevels = np.log2(levels)
                # for g,df_wt in df_list.items():
                #     ##
                #     fig, ax = plt.subplots()
                #     im = ax.contourf(df_wt.columns, df_wt.index, df_wt.values, contourlevels, extend='both',cmap=cmap)
                #     ax.set_title("{}-{}".format(g,wavelet), fontsize=20)
                #     ax.set_ylabel('Frequency', fontsize=18)
                #     ax.set_xlabel('Time', fontsize=18)
                #     fig.colorbar(im, orientation="vertical")
                #     plt.savefig(os.path.join(logdir_TK,'Wavelet-{}-{}.svg'.format(feature,g)))
                #     plt.close()
                #     ## Squared
                #     fig, ax = plt.subplots()
                #     im = ax.contourf(df_wt.columns, df_wt.index, df_wt.abs().values, contourlevels, extend='both',cmap=cmap)
                #     ax.set_title("{}-{}".format(g,wavelet), fontsize=20)
                #     ax.set_ylabel('Frequency', fontsize=18)
                #     ax.set_xlabel('Time', fontsize=18)
                #     fig.colorbar(im, orientation="vertical")
                #     plt.savefig(os.path.join(logdir_TK,'Wavelet_sq-{}-{}.svg'.format(feature,g)))
                #     plt.close()
