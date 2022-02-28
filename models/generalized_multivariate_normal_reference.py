import torch
from torch import nn
import torch.distributions

import torch
from torch import nn

class GeneralizedMultivariateNormalReference(nn.Module):
    def __init__(self, p, initial_log_r = None, fixed_log_r = None):

        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.p = p
        if initial_log_r == None and fixed_log_r == None:
            self.log_r = nn.Parameter(2*torch.ones([self.p]).to(self.device))
        if initial_log_r != None and fixed_log_r == None:
            assert initial_log_r.shape == torch.Size([self.p]), " initial_log_r should be of shape " + str(torch.Size([self.p])) + " but has shape " + str(initial_log_r.shape)
            self.log_r = nn.Parameter(initial_log_r.to(self.device))
        if initial_log_r == None and fixed_log_r != None:
            assert fixed_log_r.shape == torch.Size([self.p]), " fixed_log_r should be of shape " + str(torch.Size([self.p])) + " but has shape " + str(fixed_log_r.shape)
            self.log_r = fixed_log_r.to(self.device)
            self.log_r.requires_grad = False
        if initial_log_r != None and fixed_log_r != None:
            raise ValueError("Both initial and final values were specified")

        self.lr = 1e-3
        self.to(self.device)

    def sample(self, num_samples, r = None):
        if r is None:
            r = torch.exp(self.log_r)
        Y = torch.distributions.gamma.Gamma((1 / r), (torch.tensor(1 / 1.4142).to(self.device) ** r)).sample([num_samples]) ** (1 / r)
        u = torch.distributions.uniform.Uniform(torch.zeros(self.p).to(self.device), torch.ones(self.p).to(self.device)).sample([num_samples])
        return (u > .5) * Y - (u < .5) * Y

    def log_density(self, samples, r = None):
        if r is None:
            r = torch.exp(self.log_r)
        log_2_sqrt_2 = 1.0397
        sqrt_2 = 1.4142
        return -((torch.abs(samples)/sqrt_2)**r).sum(-1) - torch.lgamma(1+1/r).to(self.device).sum(-1) -self.p*log_2_sqrt_2

    def get_parameters(self):
        return self.state_dict()

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)