import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from models.Abstract_Models.transport_sampler import TransportSampler
from models.Utils.reference.multivariate_normal import MultivariateNormal
from models.Utils.flow.location_scale import LocationScale
from models.Utils.weight.softmax_linear import SoftmaxLinear

from models.Utils.color_visual import *

class TMCSamplerLayer(nn.Module):
    def __init__(self,p, K, p_log_density):

        super().__init__()

        self.p = p
        self.K = K

        self.w = SoftmaxLinear(self.K, self.p)
        self.T = LocationScale(self.K, self.p)

        self.q_log_density = None
        self.p_log_density = p_log_density

        self.lr = 5e-3

    def log_v(self, x):
        z = self.T.forward(x)
        log_v = self.p_log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim=-1, keepdim=True)

    def sample_backward(self, z):
        x = self.T.backward(z)
        pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
        return torch.stack([x[i, pick[i], :] for i in range(z.shape[0])])

    def log_phi(self, z):
        x = self.T.backward(z)
        return torch.logsumexp(torch.diagonal(self.log_v(x), 0, -2, -1) + self.p_log_density(x) - self.T.log_det_J(z), dim=-1)

    def log_psi(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.q_log_density(z) + self.T.log_det_J(x),dim=-1)

class RealNVPSamplerLayer(nn.Module):
    def __init__(self,p,hidden_dim, p_log_density):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = p
        net = []
        hs = [self.p] + hidden_dim + [2*self.p]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.Tanh(),
            ])
        net.pop()
        self.net = nn.Sequential(*net)

        self.mask = [torch.cat([torch.zeros(int(self.p/2)), torch.ones(self.p - int(self.p/2))], dim = 0).to(self.device),torch.cat([torch.ones(int(self.p/2)), torch.zeros(self.p - int(self.p/2))], dim = 0).to(self.device)]
        self.q_log_density = None
        self.p_log_density = p_log_density
        self.lr = 5e-5
        self.to(self.device)

    def log_phi(self,z):
        x = z
        log_det = torch.zeros(z.shape[:-1]).to(self.device)
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]*(1 - mask),out[...,self.p:]* (1 - mask)
            x = (x*(1-mask) -m)/torch.exp(log_s) + x*mask
            log_det -= torch.sum(log_s, dim=-1)
        return self.p_log_density(x) + log_det

    def sample_backward(self, z):
        x = z
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            x = (x*(1-mask) -m)/torch.exp(log_s) + x*mask
        return x

    def log_psi(self, x):
        z = x
        log_det = torch.zeros(x.shape[:-1]).to(self.device)
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            z = z*(1 - mask) * torch.exp(log_s) + m + mask * z
            log_det += torch.sum(log_s, dim = -1)
        return self.q_log_density(z) + log_det

class MixedModelSampler(TransportSampler):
    def __init__(self, target_log_density, p,structure, initial_reference=None):

        super().__init__(target_log_density,p)

        self.structure = structure
        self.N = len(self.structure)

        if initial_reference == None:
            self.reference = MultivariateNormal(self.p)
        else:
            self.reference = initial_reference

        self.para_dict = [{'params':self.reference.parameters(), 'lr':5e-3}]

        self.model = [structure[0][0](self.p,self.structure[0][1], p_log_density=self.target_log_density).to(self.device)]
        self.para_dict.insert(-1,{'params':self.model[0].parameters(), 'lr': self.model[0].lr})
        for i in range(1,self.N):
            self.model.append(structure[i][0](self.p, structure[i][1], p_log_density=self.model[i-1].model_log_phi).to(self.device))
            self.para_dict.insert(-1,{'params':self.model[i].parameters(), 'lr': self.model[i].lr})
        for i in range(self.N-1):
            self.model[i].q_log_density = self.model[i+1].model_log_psi
        self.model[-1].q_log_density = self.reference.log_density
        self.optimizer = torch.optim.Adam(self.para_dict)

    def sample(self, num_samples):
        z = self.reference.sample(num_samples)
        for i in range(self.N - 1, -1, -1):
            z = self.model[i].sample_backward(z)
        return z

    def log_density(self, x):
        return self.model[0].log_psi(x)

    def proxy_log_density(self, z):
        return self.model[-1].log_phi(z)

    def loss(self, batch):
        return - self.proxy_log_density(batch).mean()

    def train(self, num_samples, epochs, batch_size, visual = False):
        reference_samples = self.reference.sample(num_samples)
        perm = torch.randperm(reference_samples.shape[0])
        loss_values = [torch.tensor([self.loss(reference_samples[perm][i*batch_size:min((i+1)*batch_size, num_samples)]) for i in range(int(num_samples/batch_size))]).mean().item()]
        best_loss = loss_values[0]
        best_iteration = 0
        best_parameters = self.state_dict()
        pbar = tqdm(range(epochs))
        for t in pbar:
            perm = torch.randperm(num_samples)
            for i in range(int(num_samples/batch_size)+1*(int(num_samples / batch_size)!= num_samples / batch_size)):
                self.optimizer.zero_grad()
                batch_loss = self.loss(reference_samples[perm][i*batch_size:min((i+1)*batch_size, num_samples)])
                batch_loss.backward()
                self.optimizer.step()

            iteration_loss = torch.tensor([self.loss(reference_samples[perm][i*batch_size:min((i+1)*batch_size, num_samples)]) for i in range(int(num_samples/batch_size))]).mean().item()
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
            loss_values.append(iteration_loss)
            if iteration_loss< best_loss:
                best_loss = iteration_loss
                best_iteration = t+1
                best_parameters = self.state_dict()

        self.load_state_dict(best_parameters)
        if visual:
            self.train_visual(best_loss, best_iteration, loss_values)

    def model_visual(self, num_samples = 5000, flow = True):
        if self.p == 1:
            linspace = 500
            with torch.no_grad():
                samples = self.reference.sample(num_samples)
                backward_samples = [samples]
                tt = torch.linspace(torch.min(backward_samples[0]), torch.max(backward_samples[0]), linspace).unsqueeze(1).to(self.device)
                backward_density = [torch.exp(self.model[-1].q_log_density(tt))]
                backward_linspace = [tt]
                forward_density = [torch.exp(self.model[-1].log_phi(tt))]
                for i in range(self.N - 1, -1, -1):
                    samples = self.model[i].sample_backward(samples)
                    tt = torch.linspace(torch.min(samples), torch.max(samples),linspace).unsqueeze(1).to(self.device)
                    b_density = torch.exp(self.model[i].log_psi(tt))
                    f_density = torch.exp(self.model[i].p_log_density(tt))
                    backward_samples.insert(0, samples)
                    backward_linspace.insert(0, tt)
                    backward_density.insert(0,b_density)
                    forward_density.insert(0,f_density)

            fig = plt.figure(figsize=((self.N+1)*8, 2*7))
            ax = fig.add_subplot(2,self.N+1,1)
            ax.plot(backward_linspace[0].cpu(), forward_density[0].cpu(), color='red',
                    label="Input Model density")
            ax.legend(loc = 1)
            for i in range(1, self.N):
                ax = fig.add_subplot(2, self.N + 1, i+1)
                ax.plot(backward_linspace[i].cpu(), forward_density[i].cpu(), color='magenta',
                        label="Intermediate density")
                ax.legend(loc = 1)
            ax = fig.add_subplot(2, self.N + 1, self.N+1)
            ax.plot(backward_linspace[-1].cpu(), forward_density[-1].cpu(), color='orange',
                    label="Proxy density")
            ax.legend(loc = 1)
            ax = fig.add_subplot(2,self.N+1,self.N+1+1)
            sns.histplot(backward_samples[0][:,0].cpu(),  stat="density", alpha=0.5, bins=125, color='blue',label="Model Samples")
            ax.plot(backward_linspace[0].cpu(), backward_density[0].cpu(), color='blue',
                    label="Output Model density")
            ax.legend(loc = 1)
            for i in range(1, self.N):
                ax = fig.add_subplot(2, self.N + 1, self.N+1 + i+1)
                sns.histplot(backward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='purple',label="Intermediate Samples")
                ax.plot(backward_linspace[i].cpu(), backward_density[i].cpu(), color='purple',
                        label="Intermediate density")
                ax.legend(loc = 1)
            ax = fig.add_subplot(2, self.N + 1, self.N+1 + self.N+1)
            ax.plot(backward_linspace[-1].cpu(), backward_density[-1].cpu(), color='green', label="Reference density")
            sns.histplot(backward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='green',
                         label="Reference Samples")
            if hasattr(self.reference, 'r'):
                ax.text(0.01, 0.99, 'r = ' +str(round(self.reference.r.item(), 4)),
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, fontsize = 12)
            ax.legend(loc = 1)

        elif self.p > 1 and flow == True:
            delta = 200
            with torch.no_grad():
                backward_samples = [self.reference.sample(num_samples)]
                grid = torch.cat((torch.cartesian_prod(
                    torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                   torch.max(backward_samples[0][:, 0]).item(), delta),
                    torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                   torch.max(backward_samples[0][:, 1]).item(), delta)).to(self.device),
                                  torch.mean(backward_samples[0][:, 2:], dim=0) * torch.ones(delta**2,
                                                                                             self.p - 2).to(
                                      self.device)),
                                 dim=-1)
                backward_density = [torch.exp(self.reference.log_density(grid)).reshape(delta, delta).T.cpu().detach()]
                forward_density = [torch.exp(self.model[-1].model_log_phi(grid)).reshape(delta, delta).T.cpu().detach()]
                x_range = [[torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item()]]
                y_range = [[torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item()]]
                for i in range(self.N - 1, -1, -1):
                    backward_samples.insert(0, self.model[i].sample_backward(backward_samples[0]))
                    grid = torch.cat((torch.cartesian_prod(
                        torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                       torch.max(backward_samples[0][:, 0]).item(), delta),
                        torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                       torch.max(backward_samples[0][:, 1]).item(), delta)).to(self.device),
                                      torch.mean(backward_samples[0][:, 2:], dim=0) * torch.ones(delta**2,
                                                                                                 self.p - 2).to(
                                          self.device)), dim=-1)
                    backward_density.insert(0, torch.exp(self.model[i].model_log_psi(grid)).reshape(delta,
                                                                                                    delta).T.cpu().detach())
                    forward_density.insert(0, torch.exp(self.model[i].p_log_density(grid)).reshape(delta,
                                                                                                   delta).T.cpu().detach())
                    x_range.insert(0, [torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item()])
                    y_range.insert(0, [torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item()])

            fig = plt.figure(figsize=((self.N + 1) * 8, 3 * 7))
            ax = fig.add_subplot(3, self.N + 1, 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                         torch.max(backward_samples[0][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                         torch.max(backward_samples[0][:, 1]).item(), delta), forward_density[0],
                          cmap=red_cmap, shading='auto')
            ax.set_xlim((x_range[0][0], x_range[0][1]))
            ax.set_ylim((y_range[0][0], y_range[0][1]))
            ax.set_title(r'$P$ density')
            for i in range(1, self.N):
                ax = fig.add_subplot(3, self.N + 1, i + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.set_xlim((x_range[i][0], x_range[i][1]))
                ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(),
                                             torch.max(backward_samples[i][:, 0]).item(), delta),
                              torch.linspace(torch.min(backward_samples[i][:, 1]).item(),
                                             torch.max(backward_samples[i][:, 1]).item(), delta), forward_density[i],
                              cmap=pink_cmap, shading='auto')
            ax = fig.add_subplot(3, self.N + 1, self.N + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(),
                                         torch.max(backward_samples[-1][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[-1][:, 1]).item(),
                                         torch.max(backward_samples[-1][:, 1]).item(), delta), forward_density[-1],
                          cmap=orange_cmap, shading='auto')
            ax.set_xlim((x_range[-1][0], x_range[-1][1]))
            ax.set_ylim((y_range[-1][0], y_range[-1][1]))
            ax.set_title(r'$\Phi$ Density')

            ax = fig.add_subplot(3, self.N + 1, self.N + 1 + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.scatter(backward_samples[0][:, 0].cpu(), backward_samples[0][:, 1].cpu(), alpha=0.5, color=blue_color)
            ax.set_xlim((x_range[0][0], x_range[0][1]))
            ax.set_ylim((y_range[0][0], y_range[0][1]))
            ax.set_title(r'$\Psi$ Samples')
            ax = fig.add_subplot(3, self.N + 1, 2 * (self.N + 1) + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                         torch.max(backward_samples[0][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                         torch.max(backward_samples[0][:, 1]).item(), delta), backward_density[0],
                          cmap=blue_cmap, shading='auto')
            ax.set_xlim((x_range[0][0], x_range[0][1]))
            ax.set_ylim((y_range[0][0], y_range[0][1]))
            ax.set_title(r'$\Psi$ Density')
            for i in range(1, self.N):
                ax = fig.add_subplot(3, self.N + 1, self.N + 1 + i + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(backward_samples[i][:, 0].cpu(), backward_samples[i][:, 1].cpu(), alpha=0.5,
                           color=purple_color)
                ax.set_xlim((x_range[i][0], x_range[i][1]))
                ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax = fig.add_subplot(3, self.N + 1, 2 * (self.N + 1) + i + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(),
                                             torch.max(backward_samples[i][:, 0]).item(), delta),
                              torch.linspace(torch.min(backward_samples[i][:, 1]).item(),
                                             torch.max(backward_samples[i][:, 1]).item(), delta), backward_density[i],
                              cmap=purple_cmap, shading='auto')
                ax.set_xlim((x_range[i][0], x_range[i][1]))
                ax.set_ylim((y_range[i][0], y_range[i][1]))
            ax = fig.add_subplot(3, self.N + 1, self.N + 1 + self.N + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.scatter(backward_samples[-1][:, 0].cpu(), backward_samples[-1][:, 1].cpu(), alpha=0.5,
                       color=green_color)
            if hasattr(self.reference, 'r') and self.reference.r.requires_grad == True:
                ax.text(0.01, 0.99, 'r = ' + str(list(self.reference.r.detach().cpu().numpy())),
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, fontsize=12)
            ax.set_xlim((x_range[-1][0], x_range[-1][1]))
            ax.set_ylim((y_range[-1][0], y_range[-1][1]))
            ax.set_title(r'$Q$ Samples')
            ax = fig.add_subplot(3, self.N + 1, 2 * (self.N + 1) + self.N + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(),
                                         torch.max(backward_samples[-1][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[-1][:, 1]).item(),
                                         torch.max(backward_samples[-1][:, 1]).item(), delta), backward_density[-1],
                          cmap=green_cmap, shading='auto')
            if hasattr(self.reference, 'r') and self.reference.r.requires_grad == True:
                ax.text(0.01, 0.99, 'r = ' + str(list(self.reference.r.detach().cpu().numpy())),
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, fontsize=12)
            ax.set_xlim((x_range[-1][0], x_range[-1][1]))
            ax.set_ylim((y_range[-1][0], y_range[-1][1]))
            ax.set_title(r'$Q$ Density')
