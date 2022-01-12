import torch
from torch import nn

class LocationScale(nn.Module):
    def __init__(self, K, p, initial_m = None, initial_log_s = None, fixed_m = None, fixed_log_s = None):

        super().__init__()

        self.K = K
        self.p = p

        if initial_m == None and fixed_m == None:
            self.m = nn.Parameter(torch.randn(self.K, self.p))
        elif initial_m != None and fixed_m == None:
            self.m = nn.Parameter(initial_m)
        elif initial_m == None and fixed_m != None:
            self.m = fixed_m.to(self.device)
            self.m.requires_grad = False
        elif initial_m != None and fixed_m != None:
            raise ValueError("Both initial and final values were specified")

        if initial_log_s == None and fixed_log_s == None:
            self.log_s = nn.Parameter(torch.randn(self.K, self.p))
        elif initial_log_s != None and fixed_log_s == None:
            self.log_s = nn.Parameter(initial_log_s)
        elif initial_log_s == None and fixed_log_s != None:
            self.log_s = fixed_log_s.to(self.device)
            self.log_s.requires_grad = False
        elif initial_log_s != None and fixed_log_s != None:
            raise ValueError("Both initial and final values were specified")

        self.to(self.device)

    def backward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return X * torch.exp(self.log_s).expand_as(X) + self.m.expand_as(X)

    def forward(self, z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return (Z-self.m.expand_as(Z))/torch.exp(self.log_s).expand_as(Z)

    def log_det_J(self,x):
        return -self.log_s.sum(1)

    def get_parameters(self):
        return self.state_dict()

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)