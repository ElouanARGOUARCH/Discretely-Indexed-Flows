import torch
from torch import nn

class LocationScaleFlow(nn.Module):
    def __init__(self, K, p):
        super().__init__()
        self.K = K
        self.p = p

        self.m = nn.Parameter(torch.randn(self.K, self.p))
        self.log_s = nn.Parameter(torch.zeros(self.K, self.p))

    def backward(self, z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X-self.m.expand_as(X))/torch.exp(self.log_s).expand_as(X)

    def log_det_J(self,x):
        return -self.log_s.sum(-1)
