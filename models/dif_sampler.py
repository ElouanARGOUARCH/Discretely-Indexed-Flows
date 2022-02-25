import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from models.location_scale_flow import LocationScaleFlow
from models.softmax_weight import SoftmaxWeight
from models.generalized_multivariate_normal_reference import GeneralizedMultivariateNormalReference

class DIFSampler(nn.Module):
    def __init__(self, target_log_density, p, K, initial_w=None, initial_T=None):


        self.target_log_density = target_log_density
        self.p = p
        self.K = K

        if initial_w == None:
            self.w = SoftmaxWeight(self.K, self.p, [], mode = 'Linear')
        else:
            self.w = initial_w

        if initial_T == None:
            self.T = LocationScaleFlow(self.K, self.p)
        else:
            self.T = initial_T

        self.reference = MultivariateNormalReference(self.p)

        self.para_list = list(self.w.parameters()) + list(self.T.parameters()) + list(self.reference.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)
        self.to(self.device)

    def compute_log_v(self, x):
        z = self.T.forward(x)
        log_v = self.target_log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.det_J()
        return log_v - torch.logsumexp(log_v, dim=-1, keepdim=True)

    def loss(self, z, mode ='latentMLE'):
        if mode =='latentMLE':
            return - self.proxy_log_density(z).mean()
        if mode =='RB':
            x = self.T.backward(z)
            return torch.mean(torch.sum(torch.exp(self.w.log_prob(z))*(self.model_log_density(x) - self.target_log_density(x)),dim = -1))

    def sample_model(self, num_samples):
        z = self.reference.sample(num_samples)
        x = self.T.backward(z)
        pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
        return torch.stack([x[i, pick[i], :] for i in range(num_samples)])

    def proxy_log_density(self, z):
        x = self.T.backward(z)
        return torch.logsumexp(
            torch.diagonal(self.compute_log_v(x), 0, -2, -1) + self.target_log_density(x) - self.T.det_J(), dim=-1)

    def model_log_density(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(
            torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.reference.log_density(z) + self.T.det_J(),
            dim=-1)

    def train(self, num_samples, epochs, batch_size, verbose=False, mode ='latentMLE'):
        reference_samples = self.reference.sample(num_samples)
        perm = torch.randperm(reference_samples.shape[0])
        loss_values = [torch.tensor([self.loss(reference_samples[perm][i*batch_size:min((i+1)*batch_size, num_samples)], mode) for i in range(int(num_samples/batch_size))]).mean().item()]
        best_loss = loss_values[0]
        best_iteration = 0
        best_parameters = self.state_dict()
        for t in tqdm(range(epochs)):
            perm = torch.randperm(num_samples)
            for i in range(int(num_samples / batch_size)+1*(int(num_samples / batch_size) != num_samples / batch_size)):
                self.optimizer.zero_grad()
                batch_loss = self.loss(reference_samples[perm][i * batch_size:min((i + 1) * batch_size, num_samples)], mode)
                batch_loss.backward()
                self.optimizer.step()

            iteration_loss = torch.tensor(
                [self.loss(reference_samples[perm][i * batch_size:min((i + 1) * batch_size, num_samples)], mode) for i in
                 range(int(num_samples / batch_size))]).mean().item()
            loss_values.append(iteration_loss)
            if verbose == True:
                print('iteration ' + str(t) + ' : loss = ' + str(iteration_loss))
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_iteration = t+1
                best_parameters = self.state_dict()

        self.load_state_dict(best_parameters)
        self.train_visual(best_loss, best_iteration, loss_values)

