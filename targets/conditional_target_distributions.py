import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from torch.distributions import Uniform, Normal, MultivariateNormal, Exponential

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConditionalTarget:
    def __init__(self, choice, num_samples):
        self.choices = ['Gaussian Field', "Deformed Two circles", "Moons rotation"]
        self.choice = choice
        assert self.choice in self.choices, "'" + choice + "'" + ' not implemented, please select from ' + str(
            self.choices)

        if choice == 'Gaussian Field':
            self.p = 1
            self.T_prior = torch.distributions.MultivariateNormal(torch.tensor([0.]),torch.tensor([[3.]]))
            self.target_log_density = None

            def f0(theta):
                PI = torch.acos(torch.zeros(1)).item() * 2
                thetac = theta + PI
                return (torch.sin(thetac) if 0 < thetac < 2. * PI else torch.tanh(
                    thetac * .5) * 2 if thetac < 0 else torch.tanh((thetac - 2. * PI) * .5) * 2)

            def exp_g0(theta):
                PI = torch.acos(torch.zeros(1)).item() * 2
                return (torch.tensor(0.1) + torch.exp(.5 * (theta - PI)))

            D = []
            for i in range(num_samples):
                theta_i = self.T_prior.sample()
                x_i = MultivariateNormal(torch.tensor([f0(theta_i)]), torch.tensor([[exp_g0(theta_i)]])).sample()
                D.append([theta_i, x_i])
            D = torch.tensor(D)
            self.T_samples = D[:, 0].unsqueeze(-1)
            self.X_samples = D[:, 1].unsqueeze(-1)
            self.simulator = lambda thetas: torch.cat([Normal(f0(theta), exp_g0(theta)).sample() for theta in thetas], dim = 0)

        if choice == 'Deformed Two circles':
            self.p = 2
            self.target_log_density = None
            X, y = datasets.make_circles(num_samples, factor=.5, noise=0.025)
            X = StandardScaler().fit_transform(X)
            self.T_prior = MultivariateNormal(torch.zeros(self.p).to(device), torch.eye(self.p).to(device))
            self.T_samples = self.T_prior.sample([num_samples])
            self.X_samples = torch.tensor(X)[torch.randperm(X.shape[0])].float().to(device) * self.T_samples
            self.simulator = lambda thetas: torch.cat([torch.tensor(X[torch.randperm(X.shape[0])][0]).unsqueeze(0).float()* torch.abs(theta) for theta in thetas], dim =0).to(device)

        if choice == 'Moons rotation':
            self.p = 2
            self.target_log_density = None
            X, y = datasets.make_moons(num_samples, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.T_prior = Uniform(0, 3.14159265)
            T = self.T_prior.sample([num_samples])
            self.T_samples = T.unsqueeze(-1)
            rotation_matrix = torch.zeros(num_samples, 2, 2)
            rotation_matrix[:, 0, 0], rotation_matrix[:, 0, 1], rotation_matrix[:, 1, 0], rotation_matrix[:, 1,
                                                                                          1] = torch.cos(T), torch.sin(
                T), -torch.sin(T), torch.cos(T)
            self.X_samples = (torch.tensor(X).float().unsqueeze(-2) @ rotation_matrix).squeeze(-2)
            self.simulator = lambda thetas: torch.cat([torch.tensor(X[torch.randperm(X.shape[0])][0]).unsqueeze(0).float()@ torch.tensor([[torch.cos(theta), torch.sin(
                theta)],[torch.cos(theta), -torch.sin(
                theta)]]) for theta in thetas], dim =0).to(device)


    def get_target(self):
        return self.p, self.X_samples, self.T_samples, self.T_prior, self.simulator

    def target_visual(self, num_samples=5000):
        if self.choice == "Deformed Two circles":
            fig = plt.figure(figsize=(10, 10))
            for i in range(2):
                for j in range(2):
                    theta = torch.tensor([[.75 + i * 1.25, .75 + j * 1.25]])
                    T = theta.repeat(num_samples, 1)
                    X, y = datasets.make_circles(num_samples, factor=0.5, noise=0.025)
                    X = torch.tensor(StandardScaler().fit_transform(X)).float() * T
                    ax = fig.add_subplot(2, 2, i + 2 * j + 1)
                    ax.set_xlim(-5, 5)
                    ax.set_ylim(-5, 5)
                    ax.scatter(X[:, 0], X[:, 1], color='red', alpha=.3,
                               label=self.choice + ': theta = [' + str(np.round(theta[0, 0].item(), 3)) + ',' + str(
                                   np.round(theta[0, 1].item(), 3)) + ']')
                    ax.scatter([0], [0], color='black')
                    ax.arrow(0., 0., theta[0, 0], 0., color='black', head_width=0.2, head_length=0.2)
                    ax.text(theta[0, 0] - .3, -.4, "theta_x = " + str(np.round(theta[0, 0].item(), 3)))
                    ax.arrow(0., 0., 0., theta[0, 1], color='black', head_width=0.2, head_length=0.2)
                    ax.text(-.3, theta[0, 1] + .4, "theta_y = " + str(np.round(theta[0, 1].item(), 3)))
                    ax.legend()

        if self.choice == 'Moons rotation':
            fig = plt.figure(figsize=(15, 15))
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1)
                theta = torch.tensor(3.141592 / 8 * (1 + 2 * i))
                T = theta.unsqueeze(-1)
                rotation_matrix = torch.zeros(1, 2, 2)
                rotation_matrix[0, 0, 0], rotation_matrix[0, 0, 1], rotation_matrix[0, 1, 0], rotation_matrix[
                    0, 1, 1] = torch.cos(T), torch.sin(T), -torch.sin(T), torch.cos(T)
                rotation_matrix = rotation_matrix.repeat(num_samples, 1, 1)
                X, y = datasets.make_moons(num_samples, noise=0.05)
                X = (torch.tensor(X).float().unsqueeze(-2) @ rotation_matrix).squeeze(-2)
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.scatter(X[:, 0], X[:, 1], color='red', alpha=.3,
                           label=self.choice + ' samples : theta = ' + str(np.round(theta.item(), 3)))
                ax.scatter([0], [0], color='black')
                ax.axline([0, 0], [torch.cos(theta), torch.sin(theta)], color='black', linestyle='--',
                          label='Axis Rotation with angle theta')
                ax.axline([0, 0], [1., 0.], color='black')
                ax.legend()

        if self.choice == 'Gaussian Field':
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot()
            ax.scatter(self.T_samples[:num_samples], self.X_samples[:num_samples], color='red', alpha=.4,
                       label='(x|theta) samples')
            ax.set_xlabel('theta')
            ax.set_ylabel('x')
            ax.legend()
