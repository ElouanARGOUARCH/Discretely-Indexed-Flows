import torch
from torch import nn

class MultivariateNormalReference(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.mean = torch.zeros(self.p)
        self.cov = torch.eye(self.p)
        self.distribution = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def estimate_moments(self, samples):
        self.mean = torch.mean(samples, dim = 0)
        cov = torch.cov(samples.T)
        self.cov = (cov + cov.T)/2
        self.distribution = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sample(self, num_samples):
        return self.distribution.sample([num_samples])
        #return torch.randn([num_samples, self.p])

    def log_density(self, z):
        mean = self.mean.to(z.device)
        cov = self.cov.to(z.device)
        self.distribution = torch.distributions.MultivariateNormal(mean,cov)
        return self.distribution.log_prob(z)
        #return -torch.sum(torch.square(z)/2, dim = -1) - torch.log(torch.tensor([2*torch.pi], device = z.device))*self.p/2