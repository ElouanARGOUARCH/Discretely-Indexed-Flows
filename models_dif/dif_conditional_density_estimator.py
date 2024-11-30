import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from models_dif.reference_distributions import GaussianReference, NormalReference
from models_dif.weight_functions import SoftmaxWeight

class ConditionalLocationScale(nn.Module):
    def __init__(self, K, p, d, hidden_dimensions):
        super().__init__()
        self.K = K
        self.p = p
        self.d = d

        self.network_dimensions = [self.d] + hidden_dimensions + [2*self.K*self.p]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1),nn.Tanh(),])
        network.pop()
        self.f = nn.Sequential(*network)

    def backward(self, z, theta):
        assert z.shape[:-1]==theta.shape[:-1], 'number of z samples does not match the number of theta samples'
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.p
        out = torch.reshape(self.f(theta), new_desired_size)
        m, log_s = out[...,:self.p], out[...,self.p:]
        return Z * torch.exp(log_s).expand_as(Z) + m.expand_as(Z)

    def forward(self, x, theta):
        assert x.shape[:-1]==theta.shape[:-1], 'number of x samples does not match the number of theta samples'
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.p
        out = torch.reshape(self.f(theta), new_desired_size)
        m, log_s = out[...,:self.p], out[...,self.p:]
        return (X-m.expand_as(X))/torch.exp(log_s).expand_as(X)

    def log_det_J(self,x, theta):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.p
        log_s = torch.reshape(self.f(theta), new_desired_size)[..., self.p:]
        return -log_s.sum(-1)

class ConditionalDIFDensityEstimator(nn.Module):
    def __init__(self, x_samples, theta_samples, K, hidden_dimensions):
        super().__init__()
        self.x_samples = x_samples
        self.theta_samples = theta_samples
        self.p = self.x_samples.shape[-1]
        self.d = self.theta_samples.shape[-1]
        self.num_samples = x_samples.shape[0]
        assert theta_samples.shape[0] == self.num_samples, 'number of X samples does not match the number of theta samples'
        self.K = K

        self.reference = MultivariateNormalReference(self.p)

        self.w = SoftmaxWeight(self.K, self.p+self.d, hidden_dimensions)

        self.T = ConditionalLocationScale(self.K, self.p, self.d, hidden_dimensions)

        self.loss_values = []

    def compute_log_v(self,x, theta):
        assert x.shape[:-1] == theta.shape[:-1], 'wrong shapes'
        theta_unsqueezed = theta.unsqueeze(-2).repeat(1, self.K, 1)
        z = self.T.forward(x, theta)
        log_v = self.reference.log_prob(z) + torch.diagonal(self.w.log_prob(torch.cat([z, theta_unsqueezed], dim = -1)), 0, -2, -1) + self.T.log_det_J(x, theta)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x, theta):
        assert x.shape[:-1] == theta.shape[:-1], 'wrong shapes'
        z = self.T.forward(x, theta)
        pick = Categorical(torch.exp(self.compute_log_v(x, theta))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_density(self, x, theta):
        assert x.shape[:-1] == theta.shape[:-1], 'wrong shapes'
        desired_size = list(theta.shape)
        desired_size.insert(-1, self.K)
        theta_unsqueezed = theta.unsqueeze(-2).expand(desired_size)
        z = self.T.forward(x, theta)
        return torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(torch.cat([z, theta_unsqueezed], dim = -1)), 0, -2, -1)+ self.T.log_det_J(x, theta),dim=-1)

    def sample_model(self, theta):
        z = self.reference.sample(theta.shape[0])
        x = self.T.backward(z, theta)
        pick = Categorical(torch.exp(self.w.log_prob(torch.cat([z, theta], dim = -1)))).sample()
        return x[range(x.shape[0]), pick, :]
        #return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def loss(self, batch_x, batch_theta):
        batch_theta_unsqueezed = batch_theta.unsqueeze(-2).repeat(1, self.K, 1)
        z = self.T.forward(batch_x, batch_theta)
        return -torch.mean(torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(torch.cat([z, batch_theta_unsqueezed], dim = -1)), 0, -2, -1) + self.T.log_det_J(batch_x, batch_theta), dim=-1))

    def train(self, epochs, batch_size = None):

        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)

        if batch_size is None:
            batch_size = self.x_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.x_samples, self.theta_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x, theta = batch[0].to(device), batch[1].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(x, theta)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device), batch[1].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
        self.to(torch.device('cpu'))