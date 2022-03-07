import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from models.location_scale_flow import LocationScaleFlow
from models.softmax_weight import SoftmaxWeight
from models.generalized_multivariate_normal_reference import GeneralizedMultivariateNormalReference

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import seaborn as sns
import pandas as pd

class DIFDensityEstimator(nn.Module):
    def __init__(self,target_samples,K):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K

        self.reference = GeneralizedMultivariateNormalReference(self.p)

        self.w = SoftmaxWeight(self.K, self.p, [])

        self.T = LocationScaleFlow(self.K, self.p)
        self.T.m = nn.Parameter(self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])])

        self.loss_values = []

    def compute_log_v(self,x):
        with torch.no_grad():
            z = self.T.forward(x)
            log_v = self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
            return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        with torch.no_grad():
            z = self.T.forward(x)
            pick = Categorical(torch.exp(self.compute_log_v(x))).sample()
            return torch.stack([z[i,pick[i],:] for i in range(x.shape[0])])

    def log_density(self, x):
        with torch.no_grad():
            z = self.T.forward(x)
            return torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

    def sample_model(self, num_samples):
        with torch.no_grad():
            z = self.reference.sample(num_samples)
            x = self.T.backward(z)
            pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
            return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def loss(self, batch):
        z = self.T.forward(batch)
        return -torch.mean(torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(batch), dim=-1))

    def train(self, epochs, batch_size = None):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)
        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        #pbar = tqdm(range(epochs))
        #for t in pbar:
        for t in range(epochs):
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x = batch[0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(x)
                batch_loss.backward()
                self.optimizer.step()
            #with torch.no_grad():
            #    iteration_loss = torch.tensor([self.loss(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            #self.loss_values.append(iteration_loss)
            #pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))

    def train_visual(self):
        self.best_loss = min(self.loss_values)
        self.best_iteration = self.loss_values.index(self.best_loss)
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        Y1, Y2 = self.best_loss - (max(self.loss_values) - self.best_loss) / 2, max(self.loss_values) + (max(self.loss_values) - self.best_loss) / 4
        ax.set_ylim(Y1, Y2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(self.loss_values, label='Loss values during training', color='black')
        ax.scatter([self.best_iteration], [self.best_loss], color='black', marker='d')
        ax.axvline(x=self.best_iteration, ymax=(self.best_loss - self.best_loss + (max(self.loss_values) - self.best_loss) / 2) / (
                max(self.loss_values) + (max(self.loss_values) - self.best_loss) / 4 - self.best_loss + (
                max(self.loss_values) - self.best_loss) / 2), color='black', linestyle='--')
        ax.text(0, self.best_loss - (max(self.loss_values) - self.best_loss) / 8,
                'best iteration = ' + str(self.best_iteration) + '\nbest loss = ' + str(np.round(self.best_loss, 5)),
                verticalalignment='top', horizontalalignment='left', fontsize=12)
        if len(self.loss_values) > 30:
            x1, x2 = self.best_iteration - int(len(self.loss_values) / 15), min(self.best_iteration + int(len(self.loss_values) / 15),
                                                                      len(self.loss_values) - 1)
            k = len(self.loss_values) / (2.5 * (x2 - x1 + 1))
            offset = (Y2 - Y1) / (6 * k)
            y1, y2 = self.best_loss - offset, self.best_loss + offset
            axins = zoomed_inset_axes(ax, k, loc='upper right')
            axins.axvline(x=self.best_iteration, ymax=(self.best_loss - y1) / (y2 - y1), color='black', linestyle='--')
            axins.scatter([self.best_iteration], [self.best_loss], color='black', marker='d')
            axins.xaxis.set_major_locator(MaxNLocator(integer=True))
            axins.plot(self.loss_values, color='black')
            axins.set_xlim(x1 - .5, x2 + .5)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=3, loc2=4)

    def model_visual(self, num_samples = 5000):
        num_samples = min(num_samples, self.target_samples.shape[0])
        if self.p == 1:
            linspace = 500
            with torch.no_grad():
                tt = torch.linspace(torch.min(self.target_samples), torch.max(self.target_samples), linspace).unsqueeze(1)
                model_density = torch.exp(self.log_density(tt))
                model_samples = self.sample_model(num_samples)
                reference_samples = self.reference.sample(num_samples)
                tt_r = torch.linspace(torch.min(reference_samples), torch.max(reference_samples), linspace).unsqueeze(1)
                reference_density = torch.exp(self.reference.log_density(tt_r))
                proxy_samples = self.sample_latent(self.target_samples[:num_samples])
            fig = plt.figure(figsize=(28, 16))
            ax1 = fig.add_subplot(221)
            sns.histplot(self.target_samples[:, 0].cpu(),  stat="density", alpha=0.5, bins=125, color='red',
                         label="Input Target Samples")
            ax1.legend()

            ax2 = fig.add_subplot(222)
            sns.histplot(proxy_samples[:, 0].cpu(), stat='density', alpha=0.5, bins=125, color='orange',
                         label='Proxy samples')
            ax2.legend()

            ax3 = fig.add_subplot(223, sharex=ax1)
            ax3.plot(tt.cpu(), model_density.cpu(), color='blue', label="Output model density")
            sns.histplot(model_samples[:, 0].cpu(), stat='density', alpha=0.5, bins=125, color='blue',
                         label='model samples')
            ax3.legend()

            ax4 = fig.add_subplot(224, sharex=ax2)
            ax4.plot(tt_r.cpu(), reference_density.cpu(), color='green', label='reference density')
            sns.histplot(reference_samples[:, 0].cpu(), stat='density', alpha=0.5, bins=125, color='green',
                         label='Reference samples')
            if hasattr(self.reference, 'r'):
                ax4.text(0.01, 0.99, 'r = ' +str(round(self.reference.r.item(), 4)),
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax2.transAxes, fontsize = 12)
            ax4.legend()


        elif self.p > 1 and self.p<=5:
            with torch.no_grad():
                target_samples = self.target_samples[:num_samples]
                model_samples = self.sample_model(num_samples)
                reference_samples = self.reference.sample(num_samples)
                proxy_samples = self.sample_latent(target_samples)
            df_target = pd.DataFrame(target_samples.cpu().numpy())
            df_target['label']= 'Data'
            df_model = pd.DataFrame(model_samples.cpu().numpy())
            df_model['label'] = 'Model'
            df_x = pd.concat([df_target, df_model])

            df_reference = pd.DataFrame(reference_samples.cpu().numpy())
            df_reference['label'] = 'Reference'
            df_proxy = pd.DataFrame(proxy_samples.cpu().numpy())
            df_proxy['label'] = 'Proxy'
            df_z = pd.concat([df_reference, df_proxy])
            g = sns.PairGrid(df_x, hue="label", height=12 / self.p, palette= {'Data' : 'red', 'Model' : 'blue'})
            #g.map_upper(sns.scatterplot, alpha = .4)
            #g.map_lower(sns.kdeplot, levels =4)
            g.map_lower(sns.scatterplot, alpha=.5)
            g.map_diag(sns.histplot, bins = 60, alpha = .4, stat = 'density')



            g = sns.PairGrid(df_z, hue="label", height=12 / self.p, palette= {'Reference' : 'green', 'Proxy' : 'orange'})
            #g.map_upper(sns.scatterplot, alpha = .4)
            #g.map_lower(sns.kdeplot, levels =4)
            g.map_lower(sns.scatterplot, alpha=.5)
            g.map_diag(sns.histplot, bins = 60, alpha = .4, stat = 'density')
            if hasattr(self.reference, 'r'):
                def write_power_factor(*args, **kwargs):
                    ax = plt.gca()
                    id_dim = ax.get_subplotspec().rowspan.start
                    _pf = self.reference.r[id_dim]
                    label = f"r={_pf:.2f}"
                    ax.annotate(label, xy=(0.1, 0.95), size=8, xycoords=ax.transAxes)
                g.map_diag(write_power_factor)

        else:
            dim_dsplayed = 5
            perm = torch.randperm(self.p)
            with torch.no_grad():
                target_samples = self.target_samples[:num_samples]
                model_samples = self.sample_model(num_samples)
                reference_samples = self.reference.sample(num_samples)
                proxy_samples = self.sample_latent(target_samples)
            df_target = pd.DataFrame(target_samples[:,perm][:,:dim_dsplayed].cpu().numpy())
            df_target['label']= 'Data'
            df_model = pd.DataFrame(model_samples[:,perm][:,:dim_dsplayed].cpu().numpy())
            df_model['label'] = 'Model'
            df_x = pd.concat([df_target, df_model])

            df_reference = pd.DataFrame(reference_samples[:,perm][:,:dim_dsplayed].cpu().numpy())
            df_reference['label'] = 'Reference'
            df_proxy = pd.DataFrame(proxy_samples[:,perm][:,:dim_dsplayed].cpu().numpy())
            df_proxy['label'] = 'Proxy'
            df_z = pd.concat([df_reference, df_proxy])
            g = sns.PairGrid(df_x, hue="label", height=12 / dim_dsplayed, palette= {'Data' : 'red', 'Model' : 'blue'})
            #g.map_upper(sns.scatterplot, alpha = .4)
            #g.map_lower(sns.kdeplot, levels =4)
            g.map_lower(sns.scatterplot, alpha=.5)
            g.map_diag(sns.histplot, bins = 60, alpha = .4, stat = 'density')



            g = sns.PairGrid(df_z, hue="label", height=12 / dim_dsplayed, palette= {'Reference' : 'green', 'Proxy' : 'orange'})
            #g.map_upper(sns.scatterplot, alpha = .4)
            #g.map_lower(sns.kdeplot, levels =4)
            g.map_lower(sns.scatterplot, alpha=.5)
            g.map_diag(sns.histplot, bins = 60, alpha = .4, stat = 'density')
            if hasattr(self.reference, 'r'):
                def write_power_factor(*args, **kwargs):
                    ax = plt.gca()
                    id_dim = ax.get_subplotspec().rowspan.start
                    _pf = self.reference.r[id_dim]
                    label = f"r={_pf:.2f}"
                    ax.annotate(label, xy=(0.1, 0.95), size=8, xycoords=ax.transAxes)
                g.map_diag(write_power_factor)