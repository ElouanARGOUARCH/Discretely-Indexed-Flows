import torch
from torch import nn

class GaussianReference(nn.Module):
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
        return self.distribution.sample(num_samples)

    def log_prob(self, z):
        mean = self.mean.to(z.device)
        cov = self.cov.to(z.device)
        self.distribution = torch.distributions.MultivariateNormal(mean,cov)
        return self.distribution.log_prob(z)

class NormalReference(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def sample(self, num_samples):
        shape = num_samples.append(self.p)
        return torch.randn(shape)

    def log_prob(self, z):
        return -torch.sum(torch.square(z), dim = -1)/2 - self.p*torch.log(torch.tensor(2 * torch.pi))/2