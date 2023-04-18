import torch
from torch import nn
from torch.autograd import Variable

from .layer import final_hidden

class LSTM1(nn.Module):
    def __init__(self, config):
        super(LSTM1, self).__init__()
        self.config = config

        ## From GPT-2 model
        if config.hidden_size%2 != 0:
            raise ValueError('Hidden size must be specified')
        self.embed_dim  = int(config.hidden_size/2)
        self.wte_mode   = config.wte_mode
        self.vocab_size = config.vocab_size
        self.output     = config.output
        self.n_layer = config.n_layer #number of layers
        self.seq_length = config.n_positions #sequence length

        # Set word embedding method
        if self.wte_mode == 'identity':
            if self.vocab_size != self.embed_dim:
                raise Exception('identity for wte_mode cannot be used as vocab_size and embed_dim do not match.')
            self.wte = nn.Identity()
        elif self.wte_mode == 'linear':
            self.wte = nn.Linear(config.vocab_size, self.embed_dim)
        elif self.wte_mode == 'custom':
            raise Exception('custom not available yet')
        elif self.wte_mode == 'Embedding':
            self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        else:
            raise ValueError('Input valid wte_mode [identity, linear, Embedding]')

        # Create modules
        self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.embed_dim,
                            num_layers=self.n_layer, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(self.embed_dim, self.embed_dim*4) #fully connected 1
        self.fc_final = nn.Linear(self.embed_dim*4, self.embed_dim) #fully connected last layer
        self.ln_f = final_hidden(self.embed_dim, self.vocab_size)

        self.relu = nn.ReLU()

    def forward(self,input_ids,**kwargs):
        embed = input_ids.cuda()
        ## Initialize initial hidden/internal state to zeros
        h_0 = Variable(torch.zeros(self.n_layer, embed.size(0), self.embed_dim)).cuda() #hidden state
        c_0 = Variable(torch.zeros(self.n_layer, embed.size(0), self.embed_dim)).cuda() #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(embed, (h_0, c_0)) #lstm with input, hidden, and internal state
        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(output)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc_final(out) #Final Output
        out = self.ln_f(out)
        return (out,)
