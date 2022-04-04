import torch.nn as nn
import torch
from torch.nn import LayerNorm, MultiheadAttention,ModuleList,init
from torch.nn import functional as F
import copy
torch.autograd.set_detect_anomaly(True)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
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
        output = self.linear(output)
        return output

class Client():
    def __init__(self,arg,device):
        self.arg = arg
        self.device = device
        self.encoder = TransformerEncoder(self.arg.d_data,self.arg.d_feature,self.arg.heads,1)
        self.encoder.to(self.device)
        self.criterion = nn.BCELoss()
        self.labels = []
        self.optimizer_tf = torch.optim.Adam(self.encoder.parameters(),lr=1e-4,weight_decay=0.1)#weight_decay=0.2

    def train_client(self, train):
        features = []
        for idx,(x,y) in enumerate(train):
            batch_size = x.shape[0]
            x = x.to(self.device)
            x = x.to(torch.float32)
            #y = y.to(torch.float32)
            y = y.to(self.device)
            y = y.to(torch.float32)
            self.labels.append(y)
            x = x.reshape(1, batch_size, self.arg.d_data)
            feature = self.encoder(x)
            feature = feature.reshape(batch_size, self.arg.d_feature)
            features.append(feature)
        return features

    def calculate_loss(self, scores):
        for i in range(len(scores)):
            if i == 0:
                score = scores[i]
                label = self.labels[i]
            else:
                score = torch.cat((score,scores[i]))
                label = torch.cat((label,self.labels[i]))
        loss = self.criterion(score,label)
        return loss

    def get_test_labels_scores(self,scores):
        for i in range(len(scores)):
            if i == 0:
                score = scores[i]
            else:
                score = torch.cat((score, scores[i]))
        return score

    def get_gradients(self, scores):
        for i in range(len(scores)):
            if i == 0:
                score = scores[i]
                label = self.labels[i]
            else:
                score = torch.cat((score,scores[i]))
                label = torch.cat((label,self.labels[i]))
        loss = self.criterion(score, label)
        self.loss = copy.deepcopy(loss)
        loss.backward()
        grads = {'named_grads': {}}
        for name, param in self.encoder.named_parameters():
            grads['named_grads'][name] = param.grad
        return grads, self.loss

    def model_update(self, grads):
        self.encoder.train()
        self.optimizer_tf.zero_grad()
        for k, v in self.encoder.named_parameters():
            v.grad = grads[k]
        self.optimizer_tf.step()