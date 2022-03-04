import torch.distributions
import torch
from torch import nn

class GeneralizedMultivariateNormalReference(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.log_r = nn.Parameter(torch.log(2*torch.ones([self.p])))

    def sample(self, num_samples):
        r = torch.exp(self.log_r)
        Y = torch.distributions.gamma.Gamma((1 / r), (torch.tensor(1 / 1.4142)** r)).sample([num_samples]) ** (1 / r)
        u = torch.distributions.uniform.Uniform(torch.zeros(self.p), torch.ones(self.p)).sample([num_samples])
        return (u > .5) * Y - (u < .5) * Y

    def log_density(self, samples):
        r = torch.exp(self.log_r)
        log_2_sqrt_2 = 1.0397
        sqrt_2 = 1.4142
        return -((torch.abs(samples)/sqrt_2)**r).sum(-1) - torch.lgamma(1+1/r).sum(-1) -self.p*log_2_sqrt_2
