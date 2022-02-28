import torch
from torch import nn

class LocationScaleFlow(nn.Module):
    def __init__(self, K, p, initial_m = None, initial_log_s = None, fixed_m = None, fixed_log_s = None, mode = 'diag'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.K = K
        self.mode = mode
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

        if self.mode == 'diag':
            if initial_log_s == None and fixed_log_s == None:
                self.log_s = nn.Parameter(torch.zeros(self.K, self.p))
            elif initial_log_s != None and fixed_log_s == None:
                self.log_s = nn.Parameter(initial_log_s)
            elif initial_log_s == None and fixed_log_s != None:
                self.log_s = fixed_log_s.to(self.device)
                self.log_s.requires_grad = False
            elif initial_log_s != None and fixed_log_s != None:
                raise ValueError("Both initial and final values were specified")
        elif self.mode == 'full_rank':
            if initial_log_s == None and fixed_log_s == None:
                self.L = nn.Parameter(torch.eye(self.p).unsqueeze(0).repeat(self.K,1, 1).to(self.device))
            elif initial_log_s != None and fixed_log_s == None:
                self.L = nn.Parameter(initial_log_s)
            elif initial_log_s == None and fixed_log_s != None:
                self.L = fixed_log_s.to(self.device)
                self.L.requires_grad = False
            elif initial_log_s != None and fixed_log_s != None:
                raise ValueError("Both initial and final values were specified")
        self.to(self.device)

    def apply_exp_diagonal(self, tensor):
        return torch.diag(torch.diagonal(tensor).exp()) * torch.eye(self.p) + tensor * (torch.ones(self.p, self.p) - torch.eye(self.p))

    def backward(self, z):
        if self.mode =='diag':
            desired_size = list(z.shape)
            desired_size.insert(-1, self.K)
            Z = z.unsqueeze(-2).expand(desired_size)
            return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)
        if self.mode =='full_rank':
            desired_size_Z_M = list(z.shape)
            desired_size_Z_M.insert(-1, self.K)
            desired_size_S = list(z.shape)
            desired_size_S.insert(-1, self.K)
            desired_size_S.insert(-1, self.p)
            return (self.L.expand(desired_size_S) @ (
                z.unsqueeze(-2).expand(desired_size_Z_M).unsqueeze(-1))).squeeze(-1) + self.m.expand(desired_size_Z_M)

    def forward(self, x):
        if self.mode == 'diag':
            desired_size = list(x.shape)
            desired_size.insert(-1, self.K)
            X = x.unsqueeze(-2).expand(desired_size)
            return (X-self.m.expand_as(X))/torch.exp(self.log_s).expand_as(X)
        if self.mode == 'full_rank':
            desired_size_X_M = list(x.shape)
            desired_size_X_M.insert(-1, self.K)
            desired_size_S = list(x.shape)
            desired_size_S.insert(-1, self.K)
            desired_size_S.insert(-1, self.p)
            return (torch.linalg.inv(self.L).expand(desired_size_S) @ (
                (x.unsqueeze(-2).expand(desired_size_X_M) - self.m.expand(desired_size_X_M)).unsqueeze(-1))).squeeze(-1)

    def log_det_J(self,x):
        if self.mode == 'diag':
            return -self.log_s.sum(-1)
        elif self.mode =='full_rank':
            return -torch.log(torch.linalg.det(self.L))

    def get_parameters(self):
        return self.state_dict()

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)