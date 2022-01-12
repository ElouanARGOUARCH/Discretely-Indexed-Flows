import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.distributions import Categorical

from models.Abstract_Models.transport_density_estimator import TransportDensityEstimator
from models.Utils.reference.multivariate_normal import MultivariateNormal
from models.Utils.flow.location_scale import LocationScale
from models.Utils.weight.softmax_linear import SoftmaxLinear
from models.Utils.color_visual import *

class RealNVPDensityEstimatorLayer(nn.Module):
    def __init__(self,p,hidden_dim, q_log_density):

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
        self.q_log_density = q_log_density
        self.lr = 5e-5
        self.to(self.device)

    def sample_forward(self,x):
        z = x
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]*(1 - mask),out[...,self.p:]*(1 - mask)
            z = (z*(1 - mask) * torch.exp(log_s)+m) + (mask * z)
        return z

    def sample_backward(self, z):
        x = z
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            x = ((x*(1-mask) -m)/torch.exp(log_s)) + (x*mask)
        return x

    def log_psi(self, x):
        z = x
        log_det = torch.zeros(x.shape[:-1]).to(self.device)
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            z = (z*(1 - mask)*torch.exp(log_s) + m) + (mask*z)
            log_det += torch.sum(log_s, dim = -1)
        return self.q_log_density(z) + log_det

class TMCDensityEstimatorLayer(nn.Module):
    def __init__(self,p, K, q_log_density):

        super().__init__()

        self.p = p
        self.K = K

        self.w = SoftmaxLinear(self.K, self.p)
        self.T = LocationScale(self.K, self.p)

        self.q_log_density = q_log_density
        self.lr = 5e-3

    def log_v(self,x):
        z = self.T.forward(x)
        log_v = self.q_log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_forward(self,x):
        z = self.T.forward(x)
        pick = Categorical(torch.exp(self.log_v(x))).sample()
        return torch.stack([z[i,pick[i],:] for i in range(z.shape[0])])

    def sample_backward(self, z):
        x = self.T.backward(z)
        pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(uj.shape[0])])

    def log_psi(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.q_log_density(z) + torch.diagonal(self.w.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

class MixedModelDensityEstimator(TransportDensityEstimator):
    def __init__(self, target_samples, p,structure, initial_reference=None):

        super().__init__(target_samples,p)

        self.structure = structure
        self.N = len(self.structure)

        if initial_reference == None:
            self.reference = MultivariateNormal(self.p)
        else:
            self.reference = initial_reference

        self.para_dict = [{'params':self.reference.parameters(), 'lr':self.reference.lr}]

        self.model = [structure[-1][0](self.p,self.structure[-1][1], q_log_density=self.reference.log_density).to(self.device)]
        self.para_dict.insert(0,{'params':self.model[0].parameters(), 'lr': self.model[0].lr})
        for i in range(self.N - 2, -1, -1):
            self.model.insert(0, structure[i][0](self.p, structure[i][1], q_log_density=self.model[0].model_log_psi).to(self.device))
            self.para_dict.insert(0,{'params':self.model[0].parameters(), 'lr': self.model[0].lr})
        self.optimizer = torch.optim.Adam(self.para_dict)

    def sample_model(self, num_samples):
        z = self.reference.sample(num_samples)
        for i in range(self.N - 1, -1, -1):
            z = self.model[i].sample_backward(z)
        return z

    def sample_proxy(self, x):
        for i in range(self.N):
            x = self.model[i].sample_forward(x)
        return x

    def log_density(self, x):
        return self.model[0].model_log_psi(x)

    def loss(self, batch):
        return - self.model_log_density(batch).mean()

    def model_visual(self, num_samples=5000, flow=False):
        if self.p == 1:
            linspace = 500
            with torch.no_grad():
                backward_samples = [self.reference.sample(num_samples)]
                tt = torch.linspace(torch.min(backward_samples[0]), torch.max(backward_samples[0]), linspace).unsqueeze(
                    1).to(self.device)
                backward_density = [torch.exp(self.reference.log_density(tt))]
                backward_linspace = [tt]
                for i in range(self.N - 1, -1, -1):
                    samples = self.model[i].sample_backward(backward_samples[0])
                    tt = torch.linspace(torch.min(samples), torch.max(samples), linspace).unsqueeze(1).to(self.device)
                    density = torch.exp(self.model[i].model_log_psi(tt))
                    backward_samples.insert(0, samples)
                    backward_linspace.insert(0, tt)
                    backward_density.insert(0, density)

            with torch.no_grad():
                forward_samples = [self.target_samples[:num_samples]]
                for i in range(self.N):
                    forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

            fig = plt.figure(figsize=((self.N + 1) * 8, 2 * 7))
            ax = fig.add_subplot(2, self.N + 1, 1)
            sns.histplot(forward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='red',
                         label="Input Target Samples")
            ax.legend(loc=1)
            for i in range(1, self.N):
                ax = fig.add_subplot(2, self.N + 1, i + 1)
                sns.histplot(forward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='magenta',
                             label="Intermediate Samples")
                ax.legend(loc=1)
            ax = fig.add_subplot(2, self.N + 1, self.N + 1)
            sns.histplot(forward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='orange',
                         label="Proxy Samples")
            ax.legend(loc=1)

            ax = fig.add_subplot(2, self.N + 1, self.N + 1 + 1)
            sns.histplot(backward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='blue',
                         label="Model Samples")
            ax.plot(backward_linspace[0].cpu(), backward_density[0].cpu(), color='blue',
                    label="Output Model density")
            ax.legend(loc=1)
            for i in range(1, self.N):
                ax = fig.add_subplot(2, self.N + 1, self.N + 1 + i + 1)
                sns.histplot(backward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='purple',
                             label="Intermediate Samples")
                ax.plot(backward_linspace[i].cpu(), backward_density[i].cpu(), color='purple',
                        label="Intermediate density")
                ax.legend(loc=1)
            ax = fig.add_subplot(2, self.N + 1, self.N + 1 + self.N + 1)
            ax.plot(backward_linspace[-1].cpu(), backward_density[-1].cpu(), color='green', label="Reference density")
            sns.histplot(backward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='green',
                         label="Reference Samples")
            if hasattr(self.reference, 'log_r'):
                ax.text(0.01, 0.99, 'r = ' + str(round(torch.exp(self.reference.log_r).item(), 4)),
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, fontsize=12)
            ax.legend(loc=1)

        elif self.p > 1 and self.p <= 5:
            if flow == True:
                delta = 200
                with torch.no_grad():
                    samples = self.reference.sample(num_samples)
                    backward_samples = [samples]
                    grid = torch.cartesian_prod(torch.linspace(torch.min(samples[:,0]).item(), torch.max(samples[:,0]).item(),delta),torch.linspace(torch.min(samples[:,1]).item(), torch.max(samples[:,1]).item(),delta)).to(self.device)
                    grid = torch.cat((grid, torch.mean(samples[:,2:], dim = 0)*torch.ones(grid.shape[0], self.p-2).to(self.device)), dim = -1)
                    density = torch.exp(self.reference.log_density(grid)).reshape(delta,delta).T.cpu().detach()
                    backward_density = [density]
                    x_range = [[torch.min(samples[:,0]).item(), torch.max(samples[:,0]).item()]]
                    y_range = [[torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item()]]
                    for i in range(self.N - 1, -1, -1):
                        samples = self.model[i].sample_backward(backward_samples[0])
                        backward_samples.insert(0, samples)
                        grid = torch.cartesian_prod(torch.linspace(torch.min(samples[:, 0]).item(), torch.max(samples[:, 0]).item(), delta),torch.linspace(torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item(), delta)).to(self.device)
                        grid = torch.cat((grid, torch.zeros(grid.shape[0], self.p - 2).to(self.device)), dim=-1)
                        density = torch.exp(self.model[i].model_log_psi(grid)).reshape(delta, delta).T.cpu().detach()
                        backward_density.insert(0, density)
                        x_range.insert(0,[torch.min(samples[:,0]).item(), torch.max(samples[:,0]).item()])
                        y_range.insert(0,[torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item()])

                with torch.no_grad():
                    forward_samples = [self.target_samples[:num_samples]]
                    for i in range(self.N):
                        forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

                fig = plt.figure(figsize=((self.N + 1) * 8, 3 * 7))
                ax = fig.add_subplot(3, self.N + 1, 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(forward_samples[0][:, 0].cpu(), forward_samples[0][:, 1].cpu(), alpha=0.5, color=red_color)
                ax.set_title(r'$P$ samples')
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                for i in range(1, self.N):
                    ax = fig.add_subplot(3, self.N + 1, i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.scatter(forward_samples[i][:, 0].cpu(), forward_samples[i][:, 1].cpu(), alpha=0.5,
                                color=pink_color)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax = fig.add_subplot(3, self.N + 1, self.N + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(forward_samples[-1][:, 0].cpu(), forward_samples[-1][:, 1].cpu(), alpha=0.5,
                            color=orange_color)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r'$\Phi$ Samples')

                ax = fig.add_subplot(3, self.N + 1, self.N + 1 + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(backward_samples[0][:, 0].cpu(), backward_samples[0][:, 1].cpu(), alpha=0.5, color=blue_color)
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                ax.set_title(r'$\Psi$ Samples')
                ax = fig.add_subplot(3, self.N + 1, 2*(self.N + 1) + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item(), 200),torch.linspace(torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item(), 200), backward_density[0],cmap = blue_cmap,shading='auto' )
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                ax.set_title(r'$\Psi$ Density')
                for i in range(1, self.N):
                    ax = fig.add_subplot(3, self.N + 1, self.N + 1 + i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.scatter(backward_samples[i][:, 0].cpu(), backward_samples[i][:, 1].cpu(), alpha=0.5,
                                color=purple_color)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                    ax = fig.add_subplot(3, self.N + 1, 2 * (self.N + 1) + i + 1)
                    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                    ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(), torch.max(backward_samples[i][:, 0]).item(), 200),torch.linspace(torch.min(backward_samples[i][:, 1]).item(), torch.max(backward_samples[i][:, 1]).item(), 200), backward_density[i], cmap=purple_cmap,shading='auto')

                ax = fig.add_subplot(3, self.N + 1, self.N + 1 + self.N + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(backward_samples[-1][:, 0].cpu(), backward_samples[-1][:, 1].cpu(), alpha=0.5,
                            color=green_color)
                if hasattr(self.reference, 'r') and self.reference.log_r.requires_grad == True:
                    ax.text(0.01, 0.99, 'r = ' + str(list(np.round(torch.exp(self.reference.log_r)[:2].detach().cpu().numpy(),3))),
                            verticalalignment='top', horizontalalignment='left',
                            transform=ax.transAxes, fontsize=12)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r'$Q$ samples')
                ax = fig.add_subplot(3, self.N + 1, 2*(self.N + 1) + self.N + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(), torch.max(backward_samples[-1][:, 0]).item(), 200),torch.linspace(torch.min(backward_samples[-1][:, 1]).item(), torch.max(backward_samples[-1][:, 1]).item(), 200), backward_density[-1], cmap=green_cmap,shading='auto')
                if hasattr(self.reference, 'r') and self.reference.log_r.requires_grad == True:
                    ax.text(0.01, 0.99, 'r = ' + str(list(np.round(torch.exp(self.reference.log_r)[:2].detach().cpu().numpy(),3))),
                            verticalalignment='top', horizontalalignment='left',
                            transform=ax.transAxes, fontsize=12)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r"$Q$ Density")

            else:
                num_samples = min(num_samples, self.target_samples.shape[0])
                with torch.no_grad():
                    backward_samples = [self.reference.sample(num_samples)]
                    for i in range(self.N - 1, -1, -1):
                        samples = self.model[i].sample_backward(backward_samples[0])
                        backward_samples.insert(0, samples)

                with torch.no_grad():
                    forward_samples = [self.target_samples[:num_samples]]
                    for i in range(self.N):
                        forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

                dfs = []
                for i in range(self.N + 1):
                    df_forward = pd.DataFrame(forward_samples[i].cpu().numpy())
                    df_forward['label'] = 'Forward'
                    df_backward = pd.DataFrame(backward_samples[i].cpu().numpy())
                    df_backward['label'] = 'Backward'
                    dfs.append(pd.concat([df_forward, df_backward]))

                g = sns.PairGrid(dfs[0], hue="label", height=12 / self.p,
                                 palette={'Forward': 'red', 'Backward': 'blue'})
                g.map_upper(sns.scatterplot, alpha=.4)
                g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')

                for i in range(1, self.N):
                    g = sns.PairGrid(dfs[i], hue="label", height=12 / self.p,
                                     palette={'Forward': 'magenta', 'Backward': 'purple'})
                    g.map_upper(sns.scatterplot, alpha=.4)
                    g.map_diag(sns.histplot, bins=60, kde=True, alpha=.4, stat='density')

                g = sns.PairGrid(dfs[-1], hue="label", height=12 / self.p,
                                 palette={'Backward': 'green', 'Forward': 'orange'})
                g.map_upper(sns.scatterplot, alpha=.4)
                g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')
                if hasattr(self.reference, 'log_r'):
                    def write_power_factor(*args, **kwargs):
                        ax = plt.gca()
                        id_dim = ax.get_subplotspec().rowspan.start
                        _pf = torch.exp(self.reference.log_r)[id_dim]
                        label = f"r={_pf:.2f}"
                        ax.annotate(label, xy=(0.1, 0.95), size=8, xycoords=ax.transAxes)

                    g.map_diag(write_power_factor)

        else:
            dim_displayed = 5
            perm = torch.randperm(self.p).to(self.device)
            num_samples = min(num_samples, self.target_samples.shape[0])
            with torch.no_grad():
                backward_samples = [self.reference.sample(num_samples)]
                for i in range(self.N - 1, -1, -1):
                    samples = self.model[i].sample_backward(backward_samples[0])
                    backward_samples.insert(0, samples)

            with torch.no_grad():
                forward_samples = [self.target_samples[:num_samples]]
                for i in range(self.N):
                    forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

            dfs = []
            for i in range(self.N + 1):
                df_forward = pd.DataFrame(forward_samples[i][:, perm][:, :dim_displayed].cpu().numpy())
                df_forward['label'] = 'Forward'
                df_backward = pd.DataFrame(backward_samples[i][:, perm][:, :dim_displayed].cpu().numpy())
                df_backward['label'] = 'Backward'
                dfs.append(pd.concat([df_forward, df_backward]))

            g = sns.PairGrid(dfs[0], hue="label", height=12 / dim_displayed,
                             palette={'Forward': 'red', 'Backward': 'blue'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')

            for i in range(1, self.N):
                g = sns.PairGrid(dfs[i], hue="label", height=12 / dim_displayed,
                                 palette={'Forward': 'magenta', 'Backward': 'purple'})
                g.map_upper(sns.scatterplot, alpha=.4)
                g.map_diag(sns.histplot, bins=60, kde=True, alpha=.4, stat='density')

            g = sns.PairGrid(dfs[-1], hue="label", height=12 / dim_displayed,
                             palette={'Backward': 'green', 'Forward': 'orange'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')
            if hasattr(self.reference, 'log_r'):
                def write_power_factor(*args, **kwargs):
                    ax = plt.gca()
                    id_dim = ax.get_subplotspec().rowspan.start
                    _pf = torch.exp(self.reference.log_r)[id_dim]
                    label = f"r={_pf:.2f}"
                    ax.annotate(label, xy=(0.1, 0.95), size=8, xycoords=ax.transAxes)

                g.map_diag(write_power_factor)