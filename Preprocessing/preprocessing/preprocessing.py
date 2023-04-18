## System
import os

## Math
import numpy as np
import pandas as pd

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

## For color gradient plotting
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_3d(ax,x,y,z=None,hue=None,cmap=plt.get_cmap('jet'), num_parts=1):
    '''
    Plot 3D
    '''
    if hue is None:
        hue=z
    # generate a list of (x,y,z) points
    if z is not None:
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
    else:
        points = np.array([x,y]).transpose().reshape(-1,1,2)
    # print(points.shape)  # Out: (len(x),1,3)
    # set up a list of segments
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    # print(segs.shape)  # Out: ( len(x)-1, 2, 3 )
                      # see what we've done here -- we've mapped our (x,y,z)
                      # points to an array of segment start/end coordinates.
                      # segs[i,0,:] == segs[i-1,1,:]
    ## Add segments to collection
    if z is not None:
        # make the collection of segments
        lc = Line3DCollection(segs, cmap=cmap)
        lc.set_array(hue) # color the segments by our parameter
        ax.add_collection3d(lc)
    else:
        lc = LineCollection(segs, cmap=cmap)
        lc.set_array(hue) # color the segments by our parameter
        ax.add_collection(lc)
    ## Set axis
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    if z is not None:
        ax.set_zlim(z.min(), z.max())
    return ax


def interpolate(df, type='linear'):
    """
    Interpolate missing datas
    Input: dataframe with columns abstime, X, Y
    Output: dtaframe where X and Y has been interpolated
    ## To speed up the process, find ways to not have to loop through non-zero values
    """

    if type == 'linear':
        ## Check if absolute time is increasing by the same increment
        dt = df.abstime.values[1:] - df.abstime.values[:-1]
        num_dtime = len(np.unique(dt))
        if num_dtime != 1:
            raise Exception('Interpolation can only be performed on dataframe with consistent time increments')
        XY  = np.array([df.X.to_numpy(), df.Y.to_numpy()]).transpose()
        # Isolate the non-zero values
        # dX[i] = X[i+1]-X[i]
        # idx_delta[i] = number of steps to reach non-zero values
        idx_nonzeros = np.where(np.invert((XY == 0).all(1)))[0]
        XY_delta  = XY[idx_nonzeros[1:]] - XY[idx_nonzeros[:-1]]
        idx_delta = idx_nonzeros[1:] - idx_nonzeros[:-1]
        # zeros before encountering the first non-zero value
        interpolated = [XY[:idx_nonzeros[0]]]
        # For [x1] + [0] x n + [x2]
        # XY_delta = x1-x2
        # idx_delta = n+1
        # xy = x1
        # xy + np.linspace(0,increment,num+1)[:-1]
        interpolated += [np.linspace(0,increment,num+1)[:-1] + xy for increment,num,xy in zip(XY_delta, idx_delta, XY[idx_nonzeros[:-1]])]
        num_zeros_end = len(XY) - (idx_nonzeros[-1] + 1)
        interpolated += [np.array([XY[idx_nonzeros[-1]]]*(num_zeros_end+1))]
        XY_interpolated = np.concatenate(interpolated,axis=0)
        df.X = XY_interpolated[:,0]
        df.Y = XY_interpolated[:,1]

    return df


def remove_zeros(df):
    """
    Remove 0s in the beginning
    Only apply to the first CSV file!
    """
    start = df[(df.X != 0) & (df.Y != 0)].abstime.min()
    df_return = df[df.abstime >= start]
    return df_return


def center(df):
    """
    Center data
    """
    X0 = (df.X.max() + df.X.min())/2
    Y0 = (df.Y.max() + df.Y.min())/2
    df.X = df.X-X0
    df.Y = df.Y-Y0
    return df
