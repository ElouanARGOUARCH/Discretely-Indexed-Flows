from matplotlib import image
import numpy as np
import torch
from torch import nn

from models import LocationScaleFlow
from models import EMDensityEstimator
from models import DIFDensityEstimator
from models import SoftmaxWeightOver

rgb = image.imread("euler.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))

vector_density = grey.flatten()
vector_density = vector_density/torch.sum(vector_density)
lignes, colonnes = grey.shape
num_samples = 200000
cat = torch.distributions.Categorical(probs = vector_density)
categorical_samples = cat.sample([num_samples])
target_samples = torch.cat([(categorical_samples//colonnes).unsqueeze(-1), (categorical_samples%colonnes).unsqueeze(-1)], dim = -1) + torch.rand([num_samples,2])

num_samples = target_samples.shape[0]
epochs = 20
K = 900
initial_mu = torch.cartesian_prod(torch.linspace(0, lignes,30),torch.linspace(0, colonnes, 30))
initial_T = LocationScaleFlow(K, 2, initial_m = initial_mu, mode = 'full_rank')
EM = EMDensityEstimator(target_samples,K, initial_T = initial_T)
loss_values = EM.train(epochs,visual=True)

num_samples = target_samples.shape[0]
epochs = 100
batch_size = 20000
initial_T = EM.T
initial_w = SoftmaxWeightOver(K, 2, [10,10,10], mode = 'NN')
initial_w.f[-1].weight = nn.Parameter(torch.zeros(K, 10))
initial_w.f[-1].bias = nn.Parameter(EM.log_pi)
dif = DIFDensityEstimator(target_samples,K, initial_T= initial_T, initial_w = initial_w)
loss_values = dif.train(epochs,batch_size,visual=True)

delta = 300
grid = torch.cartesian_prod(torch.linspace(-lignes/8, 1.125*lignes,2*lignes),torch.linspace(-colonnes/8, 1.125*colonnes, 2*colonnes))
density = torch.exp(EM.log_density(grid)).reshape(2*lignes,2*colonnes).T.cpu().detach()
plt.imsave('try.jpg',torch.flip(torch.flip(density.T,[0,1]),[0,1]))