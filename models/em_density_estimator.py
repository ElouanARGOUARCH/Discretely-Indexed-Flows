import torch
from torch.distributions import Categorical
from tqdm import tqdm
from torch import nn

from models.location_scale_flow import LocationScaleFlow
from models.generalized_multivariate_normal_reference import GeneralizedMultivariateNormalReference

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import seaborn as sns
import pandas as pd

class EMDensityEstimator(nn.Module):
    def __init__(self,target_samples,K, initial_log_b = None, initial_T = None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_samples = target_samples.to(self.device)
        self.p = self.target_samples.shape[-1]
        self.K = K

        self.reference = GeneralizedMultivariateNormalReference(self.p, fixed_log_r= torch.log(2.*torch.ones(self.p)))

        if initial_log_b == None:
            self.log_pi= nn.Parameter(torch.log(torch.ones([self.K])/self.K).to(self.device))
        else:
            self.log_b = initial_log_b

        if initial_T == None:
            initial_m = self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])].to(self.device)
            initial_log_s =torch.eye(self.p).unsqueeze(0).repeat(self.K,1, 1).to(self.device)
            self.T = LocationScaleFlow(self.K, self.p, initial_m = initial_m, initial_log_s = initial_log_s, mode = 'full_rank')
        else:
            self.T = initial_T
        self.to(self.device)

    def compute_log_v(self,x):
        z = self.T.forward(x)
        unormalized_log_v = self.reference.log_density(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1)+ self.T.log_det_J(x)
        return unormalized_log_v - torch.logsumexp(unormalized_log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        with torch.no_grad():
            z = self.T.forward(x)
            pick = Categorical(torch.exp(self.compute_log_v(x))).sample()
            return torch.stack([z[i,pick[i],:] for i in range(x.shape[0])])

    def log_density(self, x):
        x = x.to(self.device)
        z = self.T.forward(x)
        return torch.logsumexp(self.reference.log_density(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1) + self.T.log_det_J(x),dim=-1)

    def sample_model(self, num_samples):
        with torch.no_grad():
            z = self.reference.sample(num_samples)
            x = self.T.backward(z)
            pick = Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
            return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def M_step(self, batch):
        v = torch.exp(self.compute_log_v(batch))
        c = torch.sum(v, dim=0)
        self.log_pi = nn.Parameter(torch.log(c) - torch.logsumexp(torch.log(c), dim = 0))
        self.T.m = nn.Parameter(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * batch.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1))
        temp = (batch.unsqueeze(1).repeat(1,self.K, 1) - self.T.m.unsqueeze(0).repeat(batch.shape[0],1,1)).unsqueeze(-1)
        temp2 = temp@torch.transpose(temp, -2,-1)
        temp3= nn.Parameter(torch.linalg.cholesky(torch.sum(v.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.p, self.p) * temp2,
                                                                      dim=0)/ c.unsqueeze(-1).unsqueeze(-1)))
        #log_transformed = self.apply_log_diagonal(temp3)
        self.T.L = nn.Parameter(temp3)

    def apply_log_diagonal(self, tensor):
        U = torch.ones(self.p, self.p) - torch.eye(self.p)
        return tensor * U + ((tensor*torch.eye(self.p)) + U).log()

    def to_cpu(self):
        cpu = torch.device('cpu')
        self.to(cpu)
        self.device = cpu
        self.reference.to(cpu)
        self.reference.device = cpu
        self.T.to(cpu)
        self.T.device = cpu
        self.w.to(cpu)
        self.w.device = cpu

    def to_gpu(self):
        gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(gpu)
        self.device = gpu
        self.reference.to(gpu)
        self.reference.device = gpu
        self.T.to(gpu)
        self.T.device = gpu
        self.w.to(gpu)
        self.w.device = gpu


    def train(self, epochs, visual = False):
        self.to_gpu()
        iteration_loss = -torch.mean(self.log_density(self.target_samples)).detach().item()
        loss_values = [iteration_loss]
        best_loss = loss_values[0]
        best_iteration = 0
        best_parameters = self.state_dict()
        pbar = tqdm(range(epochs))
        for t in pbar:
            self.M_step(self.target_samples)
            iteration_loss = -torch.mean(self.log_density(self.target_samples)).detach().item()
            loss_values.append(iteration_loss)
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_iteration = t+1
                best_parameters = self.state_dict()
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
        self.load_state_dict(best_parameters)
        if visual:
            self.train_visual(best_loss, best_iteration, loss_values)
        self.to_cpu()
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
            g.map_lower(sns.scatterplot, alpha=.5)
            g.map_diag(sns.histplot, bins = 60, alpha = .4, stat = 'density')

            g = sns.PairGrid(df_z, hue="label", height=12 / self.p, palette= {'Reference' : 'green', 'Proxy' : 'orange'})
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


