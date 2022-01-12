import torch
from torch import nn

from models.Utils.abstract.abstract_weight import Weight


class SoftmaxNN(Weight):
    def __init__(self, K, p, hidden_dimensions, mode = 'NN'):
        super().__init__()
        self.K = K
        self.p = p
        if self.mode == 'NN':
            network_dimensions = [self.p] + hidden_dimensions + [self.K]
            network = []
            for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
                network.extend([nn.Linear(h0, h1),nn.Tanh(),])
            network.pop()
            self.f = nn.Sequential(*network)
        elif self.mode == 'Linear':
            self.a = nn.Parameter(torch.randn(self.K, self.P))
            self.log_b = nn.Parameter(torch.randn(self.K, self.p))
        elif self.mode == 'Constant':
            self.log_pi = nn.Parameter(torch.randn(self.K, self.p))
        self.to(self.device)

    def log_prob(self, z):
        if self.mode =='NN':
            log_w = self.f.forward(z)
        elif self.mode == 'Linear':
            log_w = z @ self.a.T + self.log_b
        return log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)

    def unormalized_log_prob(self,z):
        if self.mode =='NN':
            return self.f.forward(z)
        elif self.mode =='Linear':
            return z @ self.a.T + self.log_b

    def get_parameters(self):
        return self.state_dict()

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)