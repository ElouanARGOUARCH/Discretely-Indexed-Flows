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
        self.cov = torch.cov(samples.T) #+ torch.eye(self.p)
        #print(torch.min(torch.linalg.eigvals(self.cov)))
        #print(torch.max(torch.linalg.eigvals(self.cov)))
        print(torch.linalg.eigvals(self.cov))
        self.distribution = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sample(self, num_samples):
        return self.distribution.sample([num_samples])
        #return torch.randn([num_samples, self.p])

    def log_density(self, z):
        return self.distribution.log_prob(z)
        #return -torch.sum(torch.square(z)/2, dim = -1) - torch.log(torch.tensor([2*torch.pi], device = z.device))*self.p/2