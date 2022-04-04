import torch

def get_gradients(args, client, server, scores):
    for j in range(args.client):
        for i in range(len(scores)):
            if i == 0:
                score = scores[i]
                label = client[j].labels[i]
            else:
                score = torch.cat((score,scores[i]))
                label = torch.cat((label, client[j].labels[i]))
        loss = client[j].criterion(score, label)
        client[j].optimizer_tf.zero_grad()
        server.optimizer.zero_grad()
        loss.backward()
        grads = {'named_grads': {}}
        for name, param in client[j].encoder.named_parameters():
            grads['named_grads'][name] = param.grad
        grad_server = {'named_grads': {}}
        for name, param in server.model.named_parameters():
            grad_server['named_grads'][name] = param.grad
        return grads,grad_server

def agg_grads(args, gradients):
    total_grads = {}
    for info in gradients:
        for k, v in info['named_grads'].items():
            if k not in total_grads:
                total_grads[k] = v
            if v != None:
                total_grads[k] += v
    global_grads = {}
    for k, v in total_grads.items():
        if v != None:
            global_grads[k] = torch.div(v, args.client)
        else:
            global_grads[k] = v
    return global_grads