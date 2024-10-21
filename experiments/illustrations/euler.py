import torch
from matplotlib import image
import matplotlib
import numpy
import matplotlib.pyplot as plt
from models_dif import *
from tqdm import tqdm

class logit():
    def __init__(self, alpha = 1e-2):
        self.alpha = alpha

    def transform(self,x, alpha = None):
        assert torch.all(x<=1) and torch.all(x>=0), 'can only transform value between 0 and 1'
        if alpha is None:
            alpha = self.alpha
        return torch.logit(alpha*torch.ones_like(x) + x*(1-2*alpha))

    def inverse_transform(self, x, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return (torch.sigmoid(x)-alpha*torch.ones_like(x))/(1-2*alpha)

    def log_det(self,x, alpha = None ):
        if alpha is None:
            alpha = self.alpha
        return torch.sum(torch.log((1-2*alpha)*(torch.reciprocal(alpha*torch.ones_like(x) + x*(1-2*alpha)) + torch.reciprocal((1-alpha)*torch.ones_like(x) - x*(1-2*alpha)))), dim = -1)

def plot_2d_function(f,range = [[-10,10],[-10,10]], bins = [50,50], alpha = 0.7,show = True):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plt.pcolormesh(tt_x,tt_y,f(mesh).numpy().reshape(bins[0],bins[1]).T, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)
    if show:
        plt.show()

def plot_image_2d_points(samples, bins=(200, 200), range=None, alpha = 1.,show = True):
    assert samples.shape[-1] == 2, 'Requires 2-dimensional points'
    hist, x_edges, y_edges = numpy.histogram2d(samples[:, 0].numpy(), samples[:, 1].numpy(), bins,range)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.pcolormesh(x_edges, y_edges, hist.T, cmap=matplotlib.cm.get_cmap('viridis'),alpha=alpha, lw=0)
    if show:
        plt.show()

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class GreyScale2DImageDistribution():
    def __init__(self, file):
        self.rgb = image.imread(file)
        self.lines, self.columns = self.rgb.shape[:-1]
        self.grey = torch.tensor(rgb2gray(self.rgb))

    def plot_rgb(self):
        fig = plt.figure(figsize=(8, 12))
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.imshow(self.rgb)
        plt.show()

    def plot_grey(self):
        fig = plt.figure(figsize=(8, 12))
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.imshow(self.grey)
        plt.show()

    def sample(self, num_samples):
        #an image can be seen as a mixture of isotropic 2d uniform distribution where each component is located at the pixels and has the size of a pixel. The mixtures weights are the pixel intensities.
        #first compute mixture weights
        vector_density = self.grey.flatten()
        #normalize weights
        vector_density = vector_density / torch.sum(vector_density)
        num_samples = 500000
        #samples component assignation
        cat = torch.distributions.Categorical(probs=vector_density)
        categorical_samples = cat.sample([num_samples])
        return torch.cat([((categorical_samples % self.columns + torch.rand(num_samples)) / self.columns).unsqueeze(-1),
                                    (
                                    (1 - (categorical_samples // self.columns + torch.rand(num_samples)) / self.lines)).unsqueeze(
                                        -1)], dim=-1)

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


#Sample data according to image
distribution = GreyScale2DImageDistribution("euler.jpg")
distribution.plot_rgb()
distribution.plot_grey()
num_samples = 5000
target_samples = distribution.sample([num_samples])
lines, columns = distribution.lines, distribution.columns

#display target samples
fig = plt.figure(figsize =(8,12))
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plot_image_2d_points(target_samples, bins = (lines, columns))
plt.show()


#Apply logit transform to data - logit transforms is an invertible transformation which transforms bounded samples into unbounded samples
#not necessary to run the code
logit_transform = logit(alpha = 1e-2)
transformed_samples = logit_transform.transform(target_samples)

K = 50
EM = DiagGaussianMixtEM(target_samples,K)
EM.train(50, verbose= True)

#display pdf after EM
fig = plt.figure(figsize =(8,12))
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plot_2d_function(lambda x: torch.exp(EM.log_prob(logit_transform.transform(x)).squeeze(-1) + logit_transform.log_det(x)), bins = (lines, columns), range =([[0.,1.],[0.,1.]]))
plt.show()

initial_T = LocationScaleFlow(K,2)
initial_T.m = torch.nn.Parameter(EM.m)
initial_T.log_s = torch.nn.Parameter(EM.log_s)

initial_w = SoftmaxWeight(K, 2, [128,128,128])
initial_w.f[-1].bias = torch.nn.Parameter(EM.log_pi)
initial_w.f[-1].weight = torch.nn.Parameter(torch.zeros(K,initial_w.network_dimensions[-2]))

dif = DIFDensityEstimator(target_samples,K)
dif.T = initial_T
dif.w = initial_w

epochs = 1000
batch_size = 30000
dif.train(epochs, batch_size)