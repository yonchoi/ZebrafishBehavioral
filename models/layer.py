from torch import nn

class final_hidden(nn.Module):
    def __init__(self, input_size, output_size, output='binary'):
        super().__init__()
        self.output = output
        self.linear  = nn.Linear(input_size, output_size, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        if self.output == 'binary':
            x = self.linear(x)
            x = self.sigmoid(x)
        elif self.output == 'linear':
            x = self.linear(x)
        else:
            raise ValueError('Input valid output')
        return x
