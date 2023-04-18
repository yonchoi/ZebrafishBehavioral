# /share/dennislab/projects/zfish/multitarget_HSDs/behavior
#

import os
import numpy as np
import pandas as pd
import glob

## Custom functions
import feature_calculation as fc

from preprocessing import plot_3d
from preprocessing import remove_zeros
from preprocessing import interpolate
from preprocessing import center


import anndata

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

# ==============================================================================
# Custom functions
# ==============================================================================

def read_all_raw_files(path):
    files = glob.glob(os.path.join(path,'*_raw_*.xls'))
    dfs = [pd.read_csv(f,sep='\t') for f in files]
    dfs = [parse_data(df) for df in dfs]
    return pd.concat(dfs)

def parse_data(df):
    # Modify column names
    columns = ['abstime','time','type','Location','area','X','Y']
    columns_orig = df.columns.values
    columns_orig[:len(columns)] = columns
    df.columns = columns_orig
    return df

def filter_by_well(df):
    """
    Preprocess the time series data per well
    return dictionary mapping well/location to timeseries dataframe, subset of oriignal df
    """
    # Filter for type, keep 108
    df = df[df.type==108]
    # Groupby per location/well
    groupby = {loc: df[df.Location==loc] for loc in df.Location.unique()}
    # Remove time points conitnuous zero coordinates at the beginning (tracking hasn't kicked in yet)
    groupby = {loc: remove_zeros(df_) for loc, df_ in groupby.items()}
    print(groupby['Loc01'].abstime)
    # Change time points to same increments
    def converttime(df):
        df.abstime = np.arange(len(df))
        return df
    groupby = {loc: converttime(df_) for loc, df_ in groupby.items()}
    df_test = groupby['Loc01']
    dt = df_test.abstime.values[1:] - df_test.abstime.values[:-1]
    num_dtime = len(np.unique(dt))
    print('number of time intervals',num_dtime)
    print('unique time intervals',np.unique(dt))
    print('abstime', df_test.abstime)
    # Interpolate missing time points in the middle
    groupby = {loc: interpolate(df_,'linear') for loc, df_ in groupby.items()}
    # Center the coordinates per well
    groupby = {loc: center(df_,) for loc, df_ in groupby.items()}
    # Round to first decimal point to reduce noise
    groupby = {loc: df_.round(1) for loc, df_ in groupby.items()}
    ## Match the start and end time for all locations
    start_time = max([min(df.abstime) for df in groupby.values()])
    end_time   = min([max(df.abstime) for df in groupby.values()])
    groupby = {loc: df_[(df_.abstime >= start_time) & (df_.abstime <= end_time)] for loc, df_ in groupby.items()}
    return groupby

# ==============================================================================
# Set dir
# ==============================================================================

DATA_DIR = '/share/dennislab/projects/zfish/multitarget_HSDs/behavior/raw_data/2022.10.10_arhgap11_plate1_0mM'
OUT_DIR  = 'test'
os.makedirs(OUT_DIR,exist_ok=True)

# ==============================================================================
# Load data
# ==============================================================================

df = read_all_raw_files(DATA_DIR)

df = df.sort_values(['Location','abstime'])

groupby = filter_by_well(df)


# ==============================================================================
# Plot
# ==============================================================================

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

plt.savefig(os.path.join(OUT_DIR,'locomotion.pdf'))
plt.close()

# ==============================================================================
# Calculate features
# ==============================================================================

features = ['Distance from center','Distance','Linear speed', 'Linear acceleration',
            'Angle']

logdir_plots = os.path.join(OUT_DIR,'Feature plots')
os.makedirs(logdir_plots,exist_ok=True)
palette = {'WT':'blue',
           'HET':'green',
           'HOMO':'red'}

for T in [1]:
    for K in [1]:
        for L in [1]:
            for skip in [1]:
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
            df_feature = pd.concat(df_list)

            # Need to
            df_feature['Well'] = df_feature['loc'].map(loc2well)
            df_feature['Genotype'] = df_feature['Well'].map(well2genotype)

            g = sns.FacetGrid(df_feature, col="loc", row='Feature', hue='Genotype',
                              palette=palette, sharey=False, sharex=True)
            g.map(sns.lineplot, "abstime","Value", palette=palette)
            plt.savefig(os.path.join(logdir_TK,'TimeDomain-all_subset.svg'.format(feature)))
            plt.close()
