from matplotlib import image
import numpy as np
import torch
from torch import nn
import pickle

import sys
from pathlib import Path
sys.path.append(str((Path('..') / Path('..') / Path('..')).resolve()))
sys.path.append(str((Path('.')).resolve()))

from models import EMDensityEstimator
from models import LocationScaleFlow
from models import SoftmaxWeight
from models import GeneralizedMultivariateNormalReference
from models import DIFDensityEstimator

torch.manual_seed(0)

#Load target image
rgb = image.imread("./experiments/Figures/euler/euler.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))

#Sample data according to image
vector_density = grey.flatten()
vector_density = vector_density/torch.sum(vector_density)
lignes, colonnes = grey.shape
num_samples = 200000
cat = torch.distributions.Categorical(probs = vector_density)
categorical_samples = cat.sample([num_samples])
target_samples = torch.cat([(categorical_samples//colonnes).unsqueeze(-1), (categorical_samples%colonnes).unsqueeze(-1)], dim = -1) + torch.rand([num_samples,2])

#Save target sampels
filename = './experiments/Figures/euler/euler_samples.sav'
pickle.dump(target_samples,open(filename,'wb'))

#Run EM
linspace_x = 7
linspace_y = 7
K = linspace_x*linspace_y
initial_m = torch.cartesian_prod(torch.linspace(0, lignes,linspace_x),torch.linspace(0, colonnes, linspace_y))
EM = EMDensityEstimator(target_samples,K)
EM.mu = initial_m
epochs = 200
EM.train(epochs)

#Save em
filename = './experiments/Figures/euler/euler_em.sav'
pickle.dump(EM,open(filename,'wb'))

#Run DIF with initialization EM
epochs = 10000
batch_size = 20000
initial_T = LocationScaleFlow(K,2)
initial_T.m = nn.Parameter(EM.m)
initial_T.log_s = nn.Parameter(EM.log_s)

initial_w = SoftmaxWeight(K, 2, [64,64,64])
initial_w.f[-1].weight = nn.Parameter(torch.zeros(K, 64))
initial_w.f[-1].bias = nn.Parameter(EM.log_pi)

dif = DIFDensityEstimator(target_samples,K)
dif.T = initial_T
dif.w = initial_w
dif.train(epochs, batch_size)

#Save dif
filename = './experiments/Figures/euler/euler_dif.sav'
pickle.dump(dif,open(filename,'wb'))
