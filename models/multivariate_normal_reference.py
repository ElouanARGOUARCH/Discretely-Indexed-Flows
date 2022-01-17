import torch
from torch import nn
import torch.distributions

class MultivariateNormalReference(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = p
        self.distribution = torch.distributions.MultivariateNormal(torch.zeros(self.p).to(self.device), torch.eye(self.p).to(self.device))

        self.lr = 0

        self.to(self.device)

    def sample(self, num_samples):
        return self.distribution.sample([num_samples])

    def log_density(self, samples):
        return self.distribution.log_prob(samples)

    def estimate_moments(self, samples):
        assert samples.shape[-1]==self.p, 'Unvalid samples dimension'
        if self.p >= 2:
            cov = torch.cov(samples.T)
        else:
            cov = torch.var(samples, dim=0) * torch.eye(self.p)
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.mean(samples, dim=0), cov)

    def get_parameters(self):
        return self.state_dict()

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)