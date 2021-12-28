import random
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, dim, out_dim):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim,out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class Server():
    def __init__(self,arg,device):
        self.arg = arg
        self.device = device
        self.client_list = self.choose_client()
        self.model = MLP(self.arg.d_feature,1)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.pr_performance = []
        self.roc_performance = []

    def choose_client(self):
        return [random.randint(0,self.arg.client-1) for i in range(int(self.arg.client*self.arg.frac))]

    def train_server(self,features):
        scores = []
        for i in range(len(features)):
            feature = features[i]
            batch_size = feature.shape[0]
            #feature = feature.to(self.device)
            feature = feature.reshape(batch_size, self.arg.d_feature)
            score = self.model(feature)
            score = score.view(batch_size)
            scores.append(score)
        return scores
