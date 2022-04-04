import torch

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise for CDP-FedAVG-LS Algorithm
    """
    return torch.normal(0, sigma * s, data_shape).to(device)