import torch
from torch import nn

class SoftmaxWeightOver(nn.Module):
    def __init__(self, K, p, hidden_dimensions =[], mode = 'NN'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = K
        self.p = p
        self.mode = mode
        if self.mode == 'NN':
            network_dimensions = [self.p] + hidden_dimensions + [self.K]
            network = []
            for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
                network.extend([nn.Linear(h0, h1),nn.Tanh(),])
            network.pop()
            self.f = nn.Sequential(*network)
        elif self.mode == 'Linear':
            self.a = nn.Parameter(torch.randn(self.K, self.p))
            self.log_b = nn.Parameter(torch.randn(self.K))
        elif self.mode == 'Constant':
            self.a =torch.zeros(self.K, self.p).to(self.device)
            self.log_b = nn.Parameter(torch.randn(self.K))
        self.to(self.device)

    def log_prob(self, z):
        unormalized_log_w = self.unormalized_log_prob(z)
        return unormalized_log_w - torch.logsumexp(unormalized_log_w, dim=-1, keepdim=True)

    def unormalized_log_prob(self,z):
        if self.mode == 'NN':
            return self.f.forward(z)
        elif self.mode == 'Linear' or self.mode == 'Constant':
            return z @ self.a.T + self.log_b

    def get_parameters(self):
        return self.state_dict()

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)