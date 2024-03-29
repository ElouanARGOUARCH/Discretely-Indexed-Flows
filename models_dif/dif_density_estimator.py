import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from models_dif.multivariate_normal_reference import MultivariateNormalReference
from models_dif.location_scale_flow import LocationScaleFlow
from models_dif.softmax_weight import SoftmaxWeight

class DIFDensityEstimator(nn.Module):
    def __init__(self, target_samples, K, lr = 1e-4, weight_decay = 1e-6):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K

        self.reference = MultivariateNormalReference(self.p)
        self.reference.estimate_moments(self.target_samples)

        self.w = SoftmaxWeight(self.K, self.p, [])

        self.T = LocationScaleFlow(self.K, self.p)
        self.T.m = nn.Parameter(self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])])
        self.T.log_s = nn.Parameter(torch.log(torch.var(self.target_samples, dim = 0).unsqueeze(0).repeat(self.K,1) + 1e-6 *torch.ones(self.K, self.p))/2)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_values = []

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_log_v(self,x):
        z = self.T.forward(x)
        log_v = self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.T.forward(x)
        pick = Categorical(torch.exp(self.compute_log_v(x))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_density(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

    def sample_model(self, num_samples):
        z = self.reference.sample(num_samples)
        x = self.T.backward(z)
        pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
        return x[range(x.shape[0]), pick, :]

    def loss(self, batch):
        z = self.T.forward(batch)
        return -torch.mean(torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(batch), dim=-1))

    def train(self, epochs, batch_size = None):

        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=self.lr, weight_decay=self.weight_decay)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x = batch[0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(x)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
        self.to(torch.device('cpu'))