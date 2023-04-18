import os
import numpy as np
import pandas as pd
import scanpy as sc

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import transformers
import torch

import gc

from models.GPT2Cont import GPT2ForSequenceClassificationCont

from data.util import generate_samples

torch.cuda.is_available()

## =============================================================================

trial=11
logdir = 'trials/{}/Feature plots/T-1_K-1_L-50/'.format(trial)
logdir_tf = os.path.join(logdir,'transformer')
os.makedirs(logdir_tf,exist_ok=True)

adata = sc.read_h5ad(os.path.join(logdir,'feature_matrix.h5ad'))
sc.pp.scale(adata)

features = ['Distance','Speed', 'Acceleration', 'Angle', 'Angular speed', 'Angular acceleration']

vocab_size = adata.shape[-1]

## =============================================================================

# Set config and create model
# Vocab size does not matter as we will directly input embeds

max_seq_length = 128
GPT2kwargs = {'vocab_size'  : vocab_size,
              'n_positions' : max_seq_length,
              'n_ctx'       : max_seq_length,
              'n_embd'      : 12,
              'n_layer'     : 4,
              'n_head'      : 12,
              'head_dim'    : 8,
              'wte_mode'    : 'linear'
              }

config = transformers.GPT2Config(**GPT2kwargs)
model = GPT2ForSequenceClassificationCont(config)

# Create test input
seq_len  = max_seq_length
n_sample = 50
input,metadata,attn_mask = generate_samples(adata,n_sample,seq_len,stratify=['Location'])
input     = torch.tensor(input) # (n_batch, n_seq, n_feat/n_emb)
attn_mask = torch.tensor(attn_mask) # (n_batch,n_seq)

# input_ids = input[:10]
# outputs = model.transformer(input_ids      = input_ids,
#                             attention_mask = attn_mask[:10],
#                             output_attentions = True)


## =============================================================================
from models.trainer import ContUnsupervisedTrainer, TrainingArguments
from models.dataset import CustomDataset

CDataset = CustomDataset(input, attn_mask)

## Create trainer and train
training_args = TrainingArguments("test_trainer",
                                  num_train_epochs=5,
                                  logging_steps=100,
                                  do_eval=True,
                                  do_train=True,
                                  save_steps=1000,
                                  evaluation_strategy="epoch"
                                  )

trainer = ContUnsupervisedTrainer(
    model=model.transformer,
    args=training_args,
    train_dataset=CDataset,
    eval_dataset =CDataset,
    )

train_history = trainer.train()

#### ===========================================================================
#### Visualize train history
#### ===========================================================================
from models.util_plots import (plot_prediction,
                               plot_attention,
                               plot_generate
                               )

from models.models_util import generate

#### Generate test data

input,metadata,attn_mask = generate_samples(adata,35,seq_len,
                                            subset={'Location':'Loc05'},
                                            mode  ='sequence')
input     = torch.tensor(input) # (n_batch, n_seq, n_feat/n_emb)
attn_mask = torch.tensor(attn_mask) # (n_batch,n_seq)
CDataset = CustomDataset(input, attn_mask)
inputs = CDataset[:]

#### Plot predictions

outputs = model.transformer(**inputs, output_attentions=True)

g = plot_prediction(inputs,outputs,features,mode='all')
plt.savefig(os.path.join(logdir_tf,'Train_vs_Test_all.svg'))
plt.close()

for ii in range(outputs[0].shape[0]):
    #
    g = plot_prediction(inputs,outputs,features,mode=ii)
    plt.savefig(os.path.join(logdir_tf,'Train_vs_Test-{}.svg'.format(ii)))
    plt.close()
    #
    g = plot_attention(outputs,mode=ii)
    plt.savefig(os.path.join(logdir_tf,'Attention-{}.png'.format(ii)))
    plt.close()

#### Plot generations
pred = generate(model.transformer,inputs,100)

for ii,(input_,pred_) in enumerate(zip(input.detach().numpy(),pred)):
    _ = plot_generate(np.array([input_]),np.array([pred_]),features,mode='first')
    plt.savefig(os.path.join(logdir_tf,'Generate-{}.svg'.format(ii+1)))
    plt.close()
