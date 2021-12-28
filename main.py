from par import Parser
from util import load_data
from model import Server,Client
import torch
import random
from sklearn.metrics import average_precision_score, roc_auc_score
import json

# evaluation
def aucPerformance(labels, mse):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    pr_performance.append(ap)
    roc_performance.append(roc_auc)
    return roc_auc, ap


if __name__ == '__main__':
    pr_performance = []
    roc_performance = []
    device = torch.device("cuda")
    arg = Parser().parse()
    loader = load_data.Loader(arg)
    client_data = loader.run()
    server = Server.Server(arg,device)
    client = []
    for i in range(arg.client):
        client.append(Client.Client(arg,device))

    for epoch in range(arg.epoch):
        #train
        loss_global = torch.zeros(1,requires_grad=True)
        loss_global = loss_global.to(device)
        #aggregation
        for i in range(arg.client):
            embedding_features = client[i].train_client(client_data[i])
            scores = server.train_server(embedding_features)
            loss_global = torch.mul(client[i].calculate_loss(scores),1/arg.client) + loss_global # in our setting, the number of local dataset is same.

        # updating by the same gradient
        server.optimizer.zero_grad()
        for i in range(arg.client):
            client[i].optimizer_tf.zero_grad()
        loss_global.backward()
        for i in range(arg.client):
            client[i].optimizer_tf.step()
        server.optimizer.step()
        print(epoch,"loss:",loss_global.item())

        #test
        rand_client_num = random.randint(0, arg.client - 1)
        embedding_features = client[rand_client_num].train_client(loader.test[rand_client_num])
        scores = client[rand_client_num].get_test_labels_scores(server.train_server(embedding_features))
        aucPerformance(loader.test_label,scores.cpu().detach().numpy())

        if epoch % 10 == 0:
            with open('pr.txt','w') as f:
                json.dump(pr_performance,f)
            with open('roc.txt','w') as f:
                json.dump(roc_performance,f)

