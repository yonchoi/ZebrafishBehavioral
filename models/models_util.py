import os
import gc

## Plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import torch

def generate(model, inputs, n_generate):
    gen = []
    with torch.no_grad():
        for _ in range(n_generate):
            outputs = model(**inputs,generate=True)
            ## Append the last element of prediction to the input
            pred = outputs[0][:,[-1]]
            # if model.config.output == 'binary':
            inputs['input_ids'] = torch.cat([inputs['input_ids'],pred],dim=1)[:,1:]
            gen.append(pred.cpu().numpy())
    return np.concatenate(gen,1)
