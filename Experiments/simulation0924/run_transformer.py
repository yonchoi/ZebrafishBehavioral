## System
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc

## Path
import numpy as np
import pandas as pd
import scanpy as sc

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

## pytorch
import transformers
import torch
# torch.cuda.is_available()

## model
from models.GPT2Cont import GPT2ForSequenceClassificationCont
from models.LSTM import LSTM1

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--logdir_data', type=str, default='out')
parser.add_argument('--logdir_tf' , type=str, default='transformer')

parser.add_argument('--window_len',type=int, default=1)
parser.add_argument('--window_overlap',type=int, default=0)

parser.add_argument('--method' , type=str, default='GPT2-classifier')
parser.add_argument('--n_embd', type=int, default=16)

args = parser.parse_args()
logdir_data = args.logdir_data
logdir_tf   = args.logdir_tf

window_len     = args.window_len
window_overlap = args.window_overlap

n_embd = args.n_embd
method   = args.method

"""
Inputs
    logdir_data: directory where all input files are stpred
    logdirtf   : directory to write all results
"""

logdir = 'Experiments/simulation0924'
logdir_data = os.path.join(logdir,'data/simulation2_2')

adata_filename = os.path.join(logdir_data,'simulation.h5ad')

logdir_tf = os.path.join(logdir_data,'transformer/test1/{}/n_embd-{}/window_overlap-{}'.format(method,
                                                                                               n_embd,
                                                                                               window_overlap))
os.makedirs(logdir_tf,exist_ok=True)

overwrite=True
scale = False

input_mode = 'fine-grained'

## =============================================================================

adata = sc.read_h5ad(adata_filename)

if scale:
    sc.pp.scale(adata)

features = adata.var.Feature.to_numpy()

if window_len != 1:
    features = ["{}-{}".format(f,i+1) for f in features for i in range(window_len)]
    features = np.array(features)

## =============================================================================
# Create custom dataset
from data.util import generate_samples
from models.dataset import CustomDataset

max_seq_length = 128

# Create input
seq_len  = max_seq_length

CDataset_dict={}
for data_type in ['Train','Eval']:
    input    = adata.X
    metadata = adata.obs
    X = input
    obs = metadata
    if data_type == 'Train':
        n_sample = 10000
    elif data_type == 'Eval':
        n_sample=1000
    if input_mode == 'fine-grained':
        # Create
        # input,metadata,attn_mask = generate_samples(input,metadata,None,window_len,stratify=['Sample'],mode='sequence')
        input,metadata,attn_mask = generate_samples(input,metadata,n_sample,seq_len,
                                                    window_len=window_len,
                                                    window_overlap=window_overlap,
                                                    stratify=['Sample'],
                                                    subset={'Train':[data_type]})
        # input = input.reshape(*input.shape[:2],-1)
    else:
        input,metadata,attn_mask = generate_samples(input,metadata,n_sample,seq_len,stratify=['Sample'])
    input     = torch.tensor(input) # (n_batch, n_seq, n_feat/n_emb)
    attn_mask = torch.tensor(attn_mask) # (n_batch,n_seq)
    CDataset = CustomDataset(input, attn_mask)
    CDataset_dict[data_type] = CDataset

vocab_size = input.shape[-1]

## =============================================================================
## Create and train model

from models.trainer import ContUnsupervisedTrainer, TrainingArguments
from models.util_plots import plot_training

model_filename = os.path.join(logdir_tf,'trained_model')

# GPT2kwargs = {'vocab_size'  : vocab_size,
#               'n_positions' : max_seq_length,
#               'n_ctx'       : max_seq_length,
#               'n_embd'      : n_embd,
#               'n_layer'     : 3,
#               'n_head'      : 2,
#               'head_dim'    : 8,
#               'wte_mode'    : 'linear',
#               'output'      : 'binary'
#               }
GPT2kwargs = {'vocab_size'  : vocab_size,
              'n_positions' : max_seq_length,
              'n_ctx'       : max_seq_length,
              'n_embd'      : 8,
              'n_layer'     : 3,
              'n_head'      : 2,
              'head_dim'    : 8,
              'wte_mode'    : 'linear',
              'output'      : 'binary',
              'embed_method': 'concat',
              }

config = transformers.GPT2Config(**GPT2kwargs)

if method == 'GPT2-classifier':
    model_f = GPT2ForSequenceClassificationCont
elif method == 'LSTM':
    model_f = LSTM1
else:
    raise ValueException('Input valid method: {}'.format(method))

## Practice set up
model = model_f(config)
model.cuda()
# input = CDataset_dict['Train'][:10]['input_ids'].cuda()
# out = model(input)

## Create trainer and train
## If model exists and do not want to overwrite, load model
if not os.path.isfile(os.path.join(model_filename,'pytorch_model.bin')) or overwrite:
    model = model_f(config)
    if method == 'GPT2-classifier':
        model = model.transformer
    model.cuda()
    # input = CDataset_dict['Train'][:10]['input_ids'].cuda()
    # out = model(input)
    logdir_train = os.path.join(logdir_tf,"test_trainer2")
    training_args = TrainingArguments(logdir_train,
                                      num_train_epochs=20,
                                      logging_steps=100,
                                      do_eval=True,
                                      do_train=True,
                                      save_steps=2000,
                                      evaluation_strategy="epoch",
                                      label_names = ['input_ids'],
                                      )
    trainer = ContUnsupervisedTrainer(
        model=model.transformer,
        args=training_args,
        train_dataset=CDataset_dict['Train'],
        eval_dataset =CDataset_dict['Eval'],
        loss = 'binary'
        )
    train_history = trainer.train()
    trainer.save_state()
    model.save_pretrained(model_filename)
    # plot loss
    g = plot_training(trainer)
    plt.savefig(os.path.join(logdir_train,'loss.svg'))
    plt.close()
else:
    model = model_f(config).from_pretrained(model_filename)
    model.cuda()


#### ===========================================================================
#### Visualize train history
#### ===========================================================================

from models.util_plots import (plot_prediction,
                               plot_attention,
                               plot_generate
                               )

from models.models_util import generate

# Set model as model.transformer for validation stage of seq generation
if method == 'GPT2-classifier':
    model = model.transformer

#### Generate test data --------------------------------------------------------

input,metadata,attn_mask = generate_samples(adata.X,adata.obs,35,seq_len,
                                            window_len=window_len,
                                            window_overlap=window_overlap,
                                            stratify=['Sample'],
                                            mode  ='sequence',
                                            subset={'Train':['Test']})
input     = torch.tensor(input).cuda() # (n_batch, n_seq, n_feat/n_emb)
attn_mask = torch.tensor(attn_mask).cuda() # (n_batch,n_seq)
CDataset = CustomDataset(input, attn_mask)
inputs = CDataset[:]

#### Plot predictions ----------------------------------------------------------

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, generate=True)

g = plot_prediction(inputs,outputs,features,mode='all')
plt.savefig(os.path.join(logdir_tf,'Train_vs_Test_all.svg'))
plt.close()

for ii in range(outputs[0].shape[0]):
    #
    g = plot_prediction(inputs,outputs,features,mode=ii)
    plt.savefig(os.path.join(logdir_tf,'Train_vs_Test-{}.svg'.format(ii)))
    plt.close()
    #
    if method != 'LSTM':
        g = plot_attention(outputs,mode=ii)
        plt.savefig(os.path.join(logdir_tf,'Attention-{}.png'.format(ii)))
        plt.close()

#### Plot generations ----------------------------------------------------------
input,metadata,attn_mask = generate_samples(adata.X,adata.obs,10,seq_len,
                                            window_len=window_len,
                                            window_overlap=window_overlap,
                                            stratify = ['Sample'],
                                            mode     = 'sequence',
                                            subset={'Train':['Test']})
input     = torch.tensor(input).cuda() # (n_batch, n_seq, n_feat/n_emb)
attn_mask = torch.tensor(attn_mask).cuda() # (n_batch,n_seq)
CDataset = CustomDataset(input, attn_mask)
inputs = CDataset[:]

pred = generate(model,inputs,100)

for ii,(input_,pred_) in enumerate(zip(input.cpu().numpy(),pred)):
    _ = plot_generate(np.array([input_]),np.array([pred_]),features,mode='first')
    plt.savefig(os.path.join(logdir_tf,'Generate-lineplot-{}.svg'.format(ii+1)))
    plt.close()
    _ = plot_generate(np.array([input_]),np.array([pred_]),features,mode='first',type='heatmap')
    plt.savefig(os.path.join(logdir_tf,'Generate-heatmap-{}.svg'.format(ii+1)))
    plt.close()

#### Custom inputs for generations ---------------------------------------------
input_dict = {}

# Late peak
input_array = []
input = np.zeros(128).astype('float32')
input[-20] = 1
input_array.append(input)
input = np.array(input_array*10)
input = torch.tensor(input.reshape(input.shape[0],-1,1))
input_dict['Late peak'] = input

# No peak
input_array = []
input = np.zeros(128).astype('float32')
input_array.append(input)
input = np.array(input_array*10)
input = torch.tensor(input.reshape(input.shape[0],-1,1))
input_dict['No peak'] = input

# Late peaks
input_array = []
input = np.zeros(128).astype('float32')
input[-25:-15] = 1
input_array.append(input)
input = np.array(input_array*10)
input = torch.tensor(input.reshape(input.shape[0],-1,1))
input_dict['Late peaks'] = input

# Multiple Peak
input_array = []
idx_valid = np.linspace(0,127,10).astype('int')

for ii in range(10):
    input = np.zeros(128).astype('float32')
    input[idx_valid[ii]] = 1
    input_array.append(input)

input = np.array(input_array)
input = torch.tensor(input.reshape(input.shape[0],-1,1))
input_dict['Multiple Peak'] = input

# Multiple Peaks
input_array = []
len_ = 10
idx_valid = np.linspace(0,127-len_,10).astype('int')

for ii in range(10):
    input = np.zeros(128).astype('float32')
    input[idx_valid[ii]:idx_valid[ii]+len_] = 1
    input_array.append(input)

input = np.array(input_array)
input = torch.tensor(input.reshape(input.shape[0],-1,1))
input_dict['Multiple Peaks'] = input

# 'Multiple Peaks Varinyg Width'
input_array = []
len_ = 10

for ii in range(len_):
    input = np.zeros(128).astype('float32')
    input[-20:-20+ii] = 1
    input_array.append(input)

input = np.array(input_array)
input = torch.tensor(input.reshape(input.shape[0],-1,1))
input_dict['Multiple Peaks Varinyg Width'] = input


logdir_generate_custom = os.path.join(logdir_tf,'generate_custom')
os.makedirs(logdir_generate_custom,exist_ok=True)
for custom, input in input_dict.items():
    input = input.cuda()
    attn_mask = torch.tensor(np.ones(128*len(input)).reshape(len(input),-1)).cuda()
    CDataset = CustomDataset(input, attn_mask)
    inputs = CDataset[:]
    pred = generate(model,inputs,100)
    for ii,(input_,pred_) in enumerate(zip(input.detach().cpu().numpy(),pred)):
        _ = plot_generate(np.array([input_]),np.array([pred_]),features,mode='first')
        plt.savefig(os.path.join(logdir_generate_custom,'Generate-Custom-{}-{}.svg'.format(custom,ii+1)))
        plt.close()
