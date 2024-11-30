import torch
from tqdm import tqdm
from models_dif.invertible_mappings import LocationScale
from models_dif.weight_functions import SoftmaxWeight
from models_dif.reference_distributions import GaussianReference, NormalReference

class DiagGaussianMixtEM(torch.nn.Module):
    def __init__(self,target_samples,K):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K
        self.log_pi = torch.log(torch.ones([self.K])/self.K)
        self.m = self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])]
        self.log_s = torch.log(torch.var(self.target_samples, dim = 0)).unsqueeze(0).repeat(self.K, 1)/2
        self.reference= torch.distributions.MultivariateNormal(torch.zeros(self.p), torch.eye(self.p))
        self.w = torch.distributions.Dirichlet(torch.ones(target_samples.shape[0])).sample()

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X - self.m.expand_as(X)) / torch.exp(self.log_s).expand_as(X)

    def backward(self,z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def log_det_J(self,x):
        return -torch.sum(self.log_s, dim = -1)

    def compute_log_v(self,x):
        z = self.forward(x)
        unormalized_log_v = self.reference.log_prob(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1)+ self.log_det_J(x)
        return unormalized_log_v - torch.logsumexp(unormalized_log_v, dim = -1, keepdim= True)

    def sample_latent(self,x, joint = False):
        z = self.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        if not joint:
            return z[range(z.shape[0]), pick, :]
        else:
            return z[range(z.shape[0]), pick, :],pick

    def log_prob(self, x):
        z = self.forward(x)
        return torch.logsumexp(self.reference.log_prob(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1) + self.log_det_J(x),dim=-1)

    def sample(self, num_samples, joint=False):
        z = self.reference.sample(num_samples)
        x = self.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        if not joint:
            return x[range(x.shape[0]), pick, :]
        else:
            return x[range(x.shape[0]), pick, :],pick

    def M_step(self, x,w):
        v = torch.exp(self.compute_log_v(x))*w.unsqueeze(-1)
        c = torch.sum(v, dim=0)
        self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim = 0)
        self.m = torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * x.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1)
        temp2 = torch.square(x.unsqueeze(1).repeat(1,self.K, 1) - self.m.unsqueeze(0).repeat(x.shape[0],1,1))
        self.log_s = torch.log(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * temp2,dim=0)/c.unsqueeze(-1))/2

    def train(self, epochs, verbose = False, trace_loss = False):
        if trace_loss:
            loss_values = []
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            self.M_step(self.target_samples, self.w)
            if verbose or trace_loss:
                loss = -torch.sum(self.log_prob(self.target_samples) * self.w).item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(loss))
            if trace_loss:
                loss_values.append(loss)
        if trace_loss:
            return loss_values

class DIFDensityEstimator(torch.nn.Module):
    def __init__(self, target_samples, K,hidden_dims = [], estimate_reference_moments = False):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K
        self.w = torch.distributions.Dirichlet(torch.ones(target_samples.shape[0])).sample()
        if estimate_reference_moments:
            self.reference_distribution = GaussianReference(self.p)
            self.reference_distribution.estimate_moments(self.target_samples)
        else:
            self.reference_distribution = NormalReference(self.p)

        self.W = SoftmaxWeight(self.K, self.p, hidden_dims)
        self.T = LocationScale(self.K, self.p)


    def EM_pretraining(self, epochs, verbose = False, num_samples = None):
        if num_samples is None:
            num_samples = self.target_samples.shape[0]
        else:
            num_samples = torch.min(torch.tensor([num_samples, self.target_samples.shape[0]]))
        em = DiagGaussianMixtEM(self.target_samples[:num_samples],self.K)
        em.train(epochs, verbose)
        self.T.m = torch.nn.Parameter(em.m)
        self.T.log_s = torch.nn.Parameter(em.log_s)
        self.W.f[-1].weight = torch.nn.Parameter(torch.zeros(self.K,self.W.network_dimensions[-2]))
        self.W.f[-1].bias = torch.nn.Parameter(em.log_pi)
        self.reference_distribution = NormalReference(self.p)


    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_log_v(self,x):
        z = self.T.forward(x)
        log_v = self.reference_log_prob(z) + torch.diagonal(self.W.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.T.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_prob(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.reference_distribution.log_prob(z) + torch.diagonal(self.W.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

    def sample(self, num_samples):
        z = self.reference_distribution.sample(num_samples)
        x = self.T.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.W.log_prob(z))).sample()
        return x[range(x.shape[0]), pick, :]

    def M_step(self, x,w):
        v = torch.exp(self.compute_log_v(x))*w.unsqueeze(-1)
        c = torch.sum(v, dim=0)
        #self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim = 0)
        self.T.m = torch.nn.Parameter(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * x.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1))
        temp = x.unsqueeze(1).repeat(1,self.K, 1) - self.T.m.unsqueeze(0).repeat(x.shape[0],1,1)
        temp2 = torch.square(temp)
        self.T.log_s = torch.nn.Parameter(torch.log(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * temp2,dim=0)/c.unsqueeze(-1))/2)

    def loss(self, x,w):
        return -torch.sum(w*self.log_prob(x))

    def train(self, epochs, batch_size = None, lr = 5e-3, weight_decay = 5e-5, verbose = False, trace_loss = False):
        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=lr, weight_decay=weight_decay)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(self.target_samples.to(device), self.w.to(device))
        if trace_loss:
            loss_values = []
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0],batch[1])
                batch_loss.backward()
                self.optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor([self.loss(batch[0],batch[1]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values