import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Uniform
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import math

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

class Target:
    def __init__(self, choice, num_samples):
        self.choices = ["Test","Hollow Circle","Orbits","Test Gaussian Dimension 1","Pyramid Dimension 1", "Test Gaussian Dimension 3", "Two circles", "Moons", "S Curve",
                        "Multimodal Example Dimension 2", "Unormalized Dimension 1", "Normalized Dimension 1",
                        "Multimodal Dimension 1","Bimodal Dimension 1", "Problematic case",
                        "Multimodal Dimension 2","Multimodal Dimension 4","Multimodal Dimension 8","Multimodal Dimension 16","Multimodal Dimension 32","Multimodal Dimension 64","Multimodal Dimension 128","Blob Dimension 64", "Blob Dimension 128"]
        self.choice = choice
        assert self.choice in self.choices, "'" + choice + "'" + ' not implemented, please select from ' + str(
            self.choices)

        if choice == "Test Gaussian Dimension 1":
            self.p = 1
            target = MultivariateNormal(torch.zeros(self.p), torch.eye(self.p))
            self.target_log_density = lambda samples: target.log_prob(samples)
            self.target_samples = target.sample([num_samples])

        if choice == "Pyramid Dimension 1":
            self.p = 1
            upper = torch.tensor([1.75,.85])
            mvn_target = Uniform(low = -upper, high = upper)
            cat = Categorical(torch.ones(len(upper)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples]).unsqueeze(-1)
            self.target_log_density = None

        if choice == "Test Gaussian Dimension 3":
            self.p = 3
            target = MultivariateNormal(torch.zeros(self.p), torch.eye(self.p))
            self.target_log_density = lambda samples: target.log_prob(samples)
            self.target_samples = target.sample([num_samples])

        if choice == 'Two circles':
            self.p = 2
            self.target_log_density = None
            X, y = datasets.make_circles(num_samples, factor=0.5, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X).float()

        if choice == 'Moons':
            self.p = 2
            self.target_log_density = None
            X, y = datasets.make_moons(num_samples, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X).float()

        if choice == 'S Curve':
            self.p = 2
            self.target_log_density = None
            X, t = datasets.make_s_curve(num_samples, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X[:,[0,2]]).float()

        if choice == "Multimodal Example Dimension 2":
            self.p = 2
            num_component = 25
            means = torch.tensor(
                [[0., 0.], [1., 0.], [-1., 0.], [2., 0.], [-2., 0.],[0., 1.], [1., 1.], [-1., 1.], [2., 1.], [-2., 1.],[0., -1.], [1., -1.], [-1., -1], [2., -1.], [-2., -1.], [0., 2.], [1., 2.], [-1., 2.], [2., 2.], [-2., 2.],[0., -2.], [1., -2.], [-1., -2.], [2., -2.], [-2., -2.]])
            covs = 0.01 * torch.eye(self.p).view(1, self.p, self.p).repeat(num_component, 1, 1)
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Hollow Circle":
            self.p = 2
            epsilon = 0.01
            theta = 2*3.1415*torch.rand([num_samples])
            samples = torch.cat([torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)], dim = -1)
            samples += torch.randn(samples.shape)*epsilon
            self.target_log_density = lambda samples: 1*(torch.abs(torch.square(samples[:,0]) + torch.square(samples[:,1]) - 1) < 0.707106*epsilon)
            self.target_samples = samples

        if choice == "Unormalized Dimension 1":
            self.p = 1
            num_component = 5
            means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-9]])
            covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]]])
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: 1.2 * mix_target.log_prob(samples.cpu()).to(cuda)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Normalized Dimension 1":
            self.p = 1
            num_component = 6
            means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-8.5],[-11.5]])
            covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]],[[1]]])
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(cuda)
            self.target_samples = mix_target.sample([num_samples]).to(cuda)

        if choice == "Test":
            self.p = 1
            num_component = 5
            means = torch.tensor([[7],[7],[7],[-3.25],[-6.75]])
            covs = torch.tensor([[[2.]],[[.5]],[[.25]], [[1]], [[1]]])
            comp = torch.tensor([2.,1.,.5,2.,2.])
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(cuda)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Multimodal Dimension 1":
            self.p = 1
            num_component = 7
            means = torch.tensor([[0.], [3.], [-3.], [6.], [-6.], [9.], [-9]])
            covs = 0.1 * torch.eye(self.p).view(1, self.p, self.p).repeat(num_component, 1, 1)
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Bimodal Dimension 1":
            self.p = 1
            num_component = 2
            means = torch.tensor([[5.], [-5.]])
            covs = torch.eye(self.p).view(1, self.p, self.p).repeat(num_component, 1, 1)
            comp = torch.tensor([1., 1.5])
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Problematic case":
            self.p = 3
            mixtures_target = 2
            batched_diag = torch.eye(self.p).view(1, self.p, self.p).repeat(mixtures_target, 1, 1)
            means_target = torch.stack([10. * torch.ones(self.p), -10. * torch.ones(self.p)])
            covs_target = 0.5 * batched_diag
            weights_target = torch.tensor([1 / mixtures_target] * mixtures_target)

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(weights_target / torch.sum(weights_target))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)

        if choice == "Multimodal Dimension 2":
            self.p = 2
            mixtures_target = 10 + 2*torch.randint(0,1,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = 2*L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 1*covs_target/2
            means_target = 2*self.p*torch.randn(mixtures_target, self.p)
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(cuda)

        if choice == "Multimodal Dimension 4":
            self.p = 4
            mixtures_target = 7 + 2*torch.randint(0,2,[1])
            L = 3*torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            means_target = 10*torch.randn(mixtures_target, self.p)
            weights_target = torch.ones(mixtures_target)

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(cuda)

        if choice == "Multimodal Dimension 8":
            self.p = 8
            mixtures_target = 4 + 2*torch.randint(0,3,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 3*covs_target/2
            means_target = self.p*torch.randn(mixtures_target, self.p)
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)

        if choice == "Multimodal Dimension 16":
            self.p = 16
            mixtures_target = 4 + 2*torch.randint(0,4,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 4*covs_target/2
            means_target = self.p*torch.randn(mixtures_target, self.p)
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)


        if choice == "Multimodal Dimension 32":
            self.p = 32
            mixtures_target = 4 + 2*torch.randint(0,5,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 5*covs_target/2
            means_target = self.p*torch.randn(mixtures_target, self.p)
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)

        if choice == "Blob Dimension 64":
            self.p = 64
            mixtures_target = 4 + 2*torch.randint(0,6,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 2*covs_target
            means_target = self.p*torch.randn(mixtures_target, self.p)/2
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)


        if choice == "Multimodal Dimension 64":
            self.p = 64
            mixtures_target = 4 + 2*torch.randint(0,6,[1])
            print(mixtures_target)
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 6*covs_target/2
            means_target = self.p*torch.randn(mixtures_target, self.p)
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)

        if choice == "Blob Dimension 128":
            self.p = 128
            mixtures_target = 4 + 2*torch.randint(0,7,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 7*covs_target
            means_target = self.p*torch.randn(mixtures_target, self.p)/2
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)

        if choice == "Orbits":
            self.p = 2
            number_planets = 7
            covs_target = 0.04*torch.eye(self.p).unsqueeze(0).repeat(number_planets,1,1)
            means_target = 2.5*torch.view_as_real(torch.pow(torch.exp(torch.tensor([2j * math.pi / number_planets])), torch.arange(0, number_planets)))
            weights_target = torch.ones(number_planets)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples.to(cpu)).to(cuda)

        if choice == "Multimodal Dimension 128":
            self.p = 128
            mixtures_target = 4 + 2*torch.randint(0,7,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 7*covs_target/2
            means_target = self.p*torch.randn(mixtures_target, self.p)
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])
            self.target_log_density = lambda samples: mix_target.log_prob(samples)

    def get_target(self):
        return self.p, self.target_log_density, self.target_samples

    def target_visual(self, num_samples = 5000):
        num_samples = min(num_samples, self.target_samples.shape[0])
        if self.p == 1:
            plt.figure(figsize=(10, 5))
            if self.target_log_density is not None:
                tt = torch.linspace(torch.min(self.target_samples), torch.max(self.target_samples), 500).unsqueeze(1)
                plt.plot(tt.cpu(), torch.exp(self.target_log_density(tt)).cpu(), label=self.choice + " density",
                     color='red')
            sns.histplot(self.target_samples[:num_samples][:, 0].cpu(), bins=150, color='red', stat='density', alpha=0.6,
                         label=self.choice + " samples")
            plt.legend()

        elif self.p > 1 and self.p<=5:
            df_x = pd.DataFrame(self.target_samples[:num_samples].cpu().numpy())
            df_x['label'] = 'Data'
            g = sns.PairGrid(df_x, hue="label", height=12 / self.p, palette={'Data': 'red'})
            g.map_diag(sns.histplot, stat='density', bins=100)
            g.map_upper(sns.scatterplot, alpha=.3)

        else:
            number_dim_displayed = 5
            perm = torch.randperm(self.p)
            df_x = pd.DataFrame(self.target_samples[:num_samples][:,perm][:,:number_dim_displayed].cpu().numpy())
            df_x['label'] = 'Data'
            g = sns.PairGrid(df_x, hue="label", height=12 / number_dim_displayed, palette={'Data': 'red'})
            g.map_diag(sns.histplot, stat='density', bins=100)
            g.map_upper(sns.scatterplot, alpha=.3)

