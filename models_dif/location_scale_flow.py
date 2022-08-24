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

class FullRankLocationScaleFlow(nn.Module):
    def __init__(self, K, p):
        super().__init__()
        self.K = K
        self.p = p

        self.chol = torch.eye(self.p).unsqueeze(0).repeat(self.K, 1,1)

        self.to(self.device)

    def backward(self, z):
        desired_size_Z_M = list(z.shape)
        desired_size_Z_M.insert(-1, self.K)
        desired_size_S = list(z.shape)
        desired_size_S.insert(-1, self.K)
        desired_size_S.insert(-1, self.p)
        return (((self.chol@self.chol.transpose(-1, -2)).expand(desired_size_S))@(z.unsqueeze(-2).expand(desired_size_Z_M).unsqueeze(-1))).squeeze(-1) + self.m.expand(desired_size_Z_M)

    def forward(self, x):
        desired_size_X_M = list(x.shape)
        desired_size_X_M.insert(-1, self.K)
        desired_size_S = list(x.shape)
        desired_size_S.insert(-1, self.K)
        desired_size_S.insert(-1, self.p)
        return ((torch.inverse(self.chol@self.chol.transpose(-1, -2)).expand(desired_size_S))@((x.unsqueeze(-2).expand(desired_size_X_M)-self.m.expand(desired_size_X_M)).unsqueeze(-1))).squeeze(-1)

    def log_det_J(self):
        S = self.chol@self.chol.transpose(-1, -2)
        chol = torch.cholesky(S)
        return -torch.log(torch.diagonal(chol,0,1,2)**2).sum(-1)
