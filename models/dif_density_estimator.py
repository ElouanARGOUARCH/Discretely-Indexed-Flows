import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from models.location_scale_flow import LocationScaleFlow
from models.softmax_weight import SoftmaxWeight
from models.multivariate_normal_reference import MultivariateNormalReference

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import seaborn as sns
import pandas as pd

class DIFDensityEstimator(nn.Module):
    def __init__(self,target_samples,K, initial_reference = None, initial_w = None, initial_T = None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_samples = target_samples.to(self.device)
        self.p = self.target_samples.shape[-1]
        self.K = K

        if initial_reference == None:
            self.reference = MultivariateNormalReference(self.p).to(self.device)
        else:
            self.reference = initial_reference

        if initial_w == None:
            self.w = SoftmaxWeight(self.K, self.p, [], mode = 'Linear').to(self.device)
        else:
            self.w = initial_w

        if initial_T == None:
            initial_log_s = torch.zeros(self.K, self.p).to(self.device)
            initial_m = self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])].to(self.device)
            self.T = LocationScaleFlow(self.K, self.p, initial_m = initial_m, initial_log_s = initial_log_s).to(self.device)
        else:
            self.T = initial_T

        self.para_list = list(self.w.parameters()) + list(self.T.parameters()) + list(self.reference.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)

        self.to(self.device)

    def compute_log_v(self,x):
        z = self.T.forward(x)
        log_v = self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.T.forward(x)
        pick = Categorical(torch.exp(self.compute_log_v(x))).sample()
        return torch.stack([z[i,pick[i],:] for i in range(x.shape[0])])

    def log_density(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

    def sample_model(self, num_samples):
        z = self.reference.sample(num_samples)
        x = self.T.backward(z)
        pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def loss(self, batch, mode = 'SGD'):
        if mode == 'SGD':
            return - self.log_density(batch).mean()
        elif mode == 'GradientEM':
            log_v = self.compute_log_v(batch).detach()
            z = self.T.forward(batch)
            log_joint = self.reference.log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(batch) - log_v
            return - torch.sum(torch.exp(log_v) * log_joint, dim = -1).mean()
        elif mode == 'GradientEM2':
            log_v = self.compute_log_v(batch).detach()
            z = self.T.forward(batch)
            unormalized_log_w = self.w.unormalized_log_prob(z)
            log_joint = self.reference.log_density(z) + self.T.log_det_J(batch) - log_v + torch.diagonal(unormalized_log_w, 0, -2, -1) - torch.logsumexp(unormalized_log_w, dim=-1).detach() - (torch.sum(torch.exp(unormalized_log_w), dim = -1) -torch.sum(torch.exp(unormalized_log_w), dim = -1).detach()) /(torch.sum(torch.exp(unormalized_log_w), dim = -1).detach())
            return - torch.sum(torch.exp(log_v) * log_joint, dim = -1).mean()

    def compare_loss(self):
        loss0 = self.loss(self.target_samples, mode = 'SGD')
        loss1 = self.loss(self.target_samples, mode='GradientEM')
        loss2 = self.loss(self.target_samples, mode='GradientEM2')
        return loss0, loss1, loss2


    def train(self, epochs, batch_size = None, visual = False, mode = 'SGD'):
        self.to(self.device)
        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        perm = torch.randperm(self.target_samples.shape[0])
        loss_values = [torch.tensor([self.loss(self.target_samples[perm][i*batch_size:min((i+1)*batch_size, self.target_samples.shape[0])]) for i in range(int(self.target_samples.shape[0]/batch_size))]).mean().item()]
        best_loss = loss_values[0]
        best_iteration = 0
        best_parameters = self.state_dict()
        pbar = tqdm(range(epochs))
        for t in pbar:
            perm = torch.randperm(self.target_samples.shape[0])
            for i in range(int(self.target_samples.shape[0] / batch_size)):
                self.optimizer.zero_grad()
                batch_loss = self.loss(self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.target_samples.shape[0])], mode = mode)
                batch_loss.backward()
                self.optimizer.step()

            iteration_loss = torch.tensor([self.loss(self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.target_samples.shape[0])]) for i
                                           in range(int(self.target_samples.shape[0] / batch_size))]).mean().item()
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
            loss_values.append(iteration_loss)
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_iteration = t+1
                best_parameters = self.state_dict()

        self.load_state_dict(best_parameters)
        if visual:
            self.train_visual(best_loss, best_iteration, loss_values)
        return loss_values

    def train_visual(self, best_loss, best_iteration, loss_values):
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        Y1, Y2 = best_loss - (max(loss_values) - best_loss) / 2, max(loss_values) + (max(loss_values) - best_loss) / 4
        ax.set_ylim(Y1, Y2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(loss_values, label='Loss values during training', color='black')
        ax.scatter([best_iteration], [best_loss], color='black', marker='d')
        ax.axvline(x=best_iteration, ymax=(best_loss - best_loss + (max(loss_values) - best_loss) / 2) / (
                max(loss_values) + (max(loss_values) - best_loss) / 4 - best_loss + (
                max(loss_values) - best_loss) / 2), color='black', linestyle='--')
        ax.text(0, best_loss - (max(loss_values) - best_loss) / 8,
                'best iteration = ' + str(best_iteration) + '\nbest loss = ' + str(np.round(best_loss, 5)),
                verticalalignment='top', horizontalalignment='left', fontsize=12)
        if len(loss_values) > 30:
            x1, x2 = best_iteration - int(len(loss_values) / 15), min(best_iteration + int(len(loss_values) / 15),
                                                                      len(loss_values) - 1)
            k = len(loss_values) / (2.5 * (x2 - x1 + 1))
            offset = (Y2 - Y1) / (6 * k)
            y1, y2 = best_loss - offset, best_loss + offset
            axins = zoomed_inset_axes(ax, k, loc='upper right')
            axins.axvline(x=best_iteration, ymax=(best_loss - y1) / (y2 - y1), color='black', linestyle='--')
            axins.scatter([best_iteration], [best_loss], color='black', marker='d')
            axins.xaxis.set_major_locator(MaxNLocator(integer=True))
            axins.plot(loss_values, color='black')
            axins.set_xlim(x1 - .5, x2 + .5)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=3, loc2=4)

    def model_visual(self, num_samples = 5000):
        num_samples = min(num_samples, self.target_samples.shape[0])
        if self.p == 1:
            linspace = 500
            with torch.no_grad():
                tt = torch.linspace(torch.min(self.target_samples), torch.max(self.target_samples), linspace).unsqueeze(1).to(self.device)
                model_density = torch.exp(self.log_density(tt))
                model_samples = self.sample_model(num_samples)
                reference_samples = self.reference.sample(num_samples)
                tt_r = torch.linspace(torch.min(reference_samples), torch.max(reference_samples), linspace).unsqueeze(1).to(self.device)
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
            perm = torch.randperm(self.p).to(self.device)
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