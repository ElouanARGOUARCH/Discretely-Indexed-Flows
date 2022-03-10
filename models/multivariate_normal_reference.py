import torch
from torch import nn

class MultivariateNormalReference(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def sample(self, num_samples):
        return torch.randn([num_samples, self.p])

    def log_density(self, z):
        return -torch.sum(torch.square(z)/2, dim = -1) - torch.log(torch.tensor([2*torch.pi], device = z.device))*self.p/2