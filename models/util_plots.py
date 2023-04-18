import os

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import torch

def plot_prediction(inputs,outputs,features=None,mode='first'):
    ## Only plot the first sample
    input = inputs['input_ids'][:,1:].detach().cpu().numpy()
    pred = outputs[0][:,:-1].detach().cpu().numpy()
    if mode == 'first':
        input = input[0]
        pred  = pred[0]
    elif mode == 'all':
        input = input.reshape(-1,input.shape[-1])
        pred  = pred.reshape(-1,input.shape[-1])
    else:
        input = input[mode]
        pred  = pred[mode]
    ##
    df_list = []
    df = pd.DataFrame(input,columns=features)
    df['Type'] = 'Orig'
    df_list.append(df)
    df = pd.DataFrame(pred,columns=features)
    df['Type'] = 'Pred'
    df_list.append(df)
    ##
    df_plot = pd.concat(df_list)
    df_plot['Seq index'] = df_plot.index
    df_plot = df_plot.melt(id_vars=['Type','Seq index'],var_name="Feature", value_name='Value')
    ##
    # g = sns.FacetGrid(df_plot, row="Feature", hue='Type')
    # g.map(sns.lineplot, x="Seq index",y='Value')
    # plt.savefig()
    g = sns.relplot(
        data=df_plot,
        x="Seq index", y="Value",
        hue="Type", row="Feature",
        kind="line",
        height=5, aspect=3, facet_kws=dict(sharey=False),
    )
    return g


def plot_attention(outputs,mode='first',cmap="rocket_r"):
    attentions = torch.stack(outputs['attentions']) # (n_layers, n_batch, ...)
    attentions = attentions.swapaxes(1,0) # (n_batch, n_layers, ...)
    if mode == 'first':
        attentions=attentions[0]
    else:
        attentions=attentions[mode]
        # attentions=attentions.reshape(-1,*attentions.shape[2:])
    palette = sns.cubehelix_palette()*5
    attentions = attentions.swapaxes(1,2)
    attn_shape = attentions.shape
    attentions = attentions.reshape(np.prod(attn_shape[:2]),
                                    np.prod(attn_shape[2:4]))
    attentions = attentions.detach().cpu().numpy()
    row_colors = np.repeat(np.arange(attn_shape[0]),attn_shape[1])
    row_colors = [palette[ii] for ii in row_colors]
    col_colors = np.repeat(np.arange(attn_shape[2]),attn_shape[3])
    col_colors = [palette[ii] for ii in col_colors]
    g = sns.clustermap(attentions,
                       cmap=cmap,
                       row_cluster=False,
                       col_cluster=False,
                       row_colors=row_colors,
                       col_colors=col_colors)
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    return g


def plot_generate(input,pred,features=None,mode='first',type='lineplot',**kwargs):
    if mode == 'first':
        input = input[0]
        pred  = pred[0]
    elif mode == 'all':
        input = input.reshape(-1,input.shape[-1])
        pred  = pred.reshape(-1,input.shape[-1])
    if type == 'lineplot':
        ##
        df_list = []
        df = pd.DataFrame(input,columns=features)
        df['Type'] = 'Orig'
        df_list.append(df)
        df = pd.DataFrame(pred,columns=features)
        df['Type'] = 'Pred'
        df_list.append(df)
        ##
        df_plot = pd.concat(df_list)
        df_plot['Seq index'] = np.arange(len(df_plot))
        df_plot = df_plot.melt(id_vars=['Type','Seq index'],var_name="Feature", value_name='Value')
        ##
        # g = sns.FacetGrid(df_plot, row="Feature", hue='Type')
        # g.map(sns.lineplot, x="Seq index",y='Value')
        # plt.savefig()
        g = sns.relplot(
            data=df_plot,
            x="Seq index", y="Value",
            hue="Type", row="Feature",
            kind="line",
            height=5, aspect=3, facet_kws=dict(sharey=False),
            **kwargs,
        )
    elif type == 'heatmap':
        x = np.concatenate([input,pred],0).transpose()
        col_colors = ['blue'] * len(input) + ['orange'] * len(pred)
        g = sns.clustermap(data=x,
                           col_cluster=False,
                           row_cluster=False,
                           col_colors=col_colors,
                           **kwargs)
    return g


def plot_training(trainer=None,log_history=None):
    log_history = pd.concat([pd.DataFrame(i,index=[0]) for i in trainer.state.log_history])
    df_list = []
    for category in ['loss','eval_loss']:
        df_ = log_history.dropna(subset=[category])
        df_add = pd.DataFrame({'Epoch'  :df_.epoch ,
                               'Step'   :df_.step ,
                               'Loss' :df_[category],
                               'Type'   :category })
        df_list.append(df_add)
    df_plot = pd.concat(df_list).reset_index()
    g = sns.lineplot(data=df_plot,x='Step',y='Loss',hue='Type')
    return g
