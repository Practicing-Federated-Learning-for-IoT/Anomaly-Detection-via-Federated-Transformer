import torch.nn as nn
from torch.nn import LayerNorm, MultiheadAttention,ModuleList,init
from torch.nn import functional as F
import copy

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    #__constants__ = ['norm']

    def __init__(self, d_model, d_feature, n_heads, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model, n_heads)
        self.layers = _get_clones(self.encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.linear = nn.Linear(d_model, d_feature)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        output = self.linear(output)
        return output


class MLP(nn.Module):
    def __init__(self, dim, out_dim):
        super(MLP,self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim,64),
            nn.Sigmoid(),
            nn.Linear(64,16),
            nn.Sigmoid(),
            nn.Linear(16,self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x