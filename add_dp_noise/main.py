from par import Parser
from utils import load_data, gaussian_noise,Update
from model import Server, Client
import torch
import json
import random
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise


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
        client.append(Client.Client(arg, device))

    for epoch in range(arg.epoch):
        gradients = []
        gradients_server = []
        loss_global = torch.zeros(1, requires_grad=True)
        loss_global = loss_global.to(device)
        for i in range(arg.client):
            embedding_features = client[i].train_client(client_data[i])
            embedding_features = torch.stack(embedding_features)
            #sigma = compute_noise(1, 0.01, 4.0, 1 * 150, 1e-3,
                                  #1e-5)  # (n, batch_size, target_epsilon, epochs, delta, noise_lbd)
            #embedding_features += gaussian_noise.gaussian_noise(embedding_features.shape, 32, sigma, device=device)/len(embedding_features)
            scores = server.train_server(embedding_features)
            ##in our setting, the number of local data is same
            gradient, gradient_server = Update.get_gradients(arg, client, server, scores)
            gradients.append(gradient)
            gradients_server.append(gradient_server)

        global_grads = Update.agg_grads(arg, gradients)
        server_grads = Update.agg_grads(arg, gradients_server)
        for i in range(arg.client):
            client[i].model_update(global_grads)
        server.model_update(server_grads)

        #updating by the same gradient
        #server.optimizer.zero_grad()
        #loss_global.backward()
        #server.optimizer.step()
        #print(epoch,"loss:",loss_global.item())
        print(epoch)


        rand_client_num = random.randint(0, arg.client - 1)
        embedding_features = client[rand_client_num].train_client(loader.test[rand_client_num])
        scores = client[rand_client_num].get_test_labels_scores(server.train_server(embedding_features))
        aucPerformance(loader.test_label,scores.cpu().detach().numpy())
        if epoch % 10 == 0:
            with open('pr.txt', 'w') as f:
                json.dump(pr_performance, f)
            with open('roc.txt', 'w') as f:
                json.dump(roc_performance, f)
