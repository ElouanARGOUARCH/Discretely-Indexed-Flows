from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import pickle

from models import LocationScaleFlow
from models import EMDensityEstimator
from models import DIFDensityEstimator
from models import SoftmaxWeight
from models import GeneralizedMultivariateNormalReference

torch.manual_seed(0)
#Load target image
rgb = image.imread("gauss.jpg")
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
filename = 'gauss_samples.sav'
pickle.dump(target_samples,open(filename,'wb'))

#Run EM
epochs = 200
K = 49
initial_m = torch.cartesian_prod(torch.linspace(0, lignes,7),torch.linspace(0, colonnes, 7))
initial_L = torch.eye(2).unsqueeze(0).repeat(K, 1, 1)
initial_T = LocationScaleFlow(K, 2, initial_m = initial_m,initial_log_s= initial_L, mode = 'full_rank')
EM = EMDensityEstimator(target_samples,K, initial_T = initial_T)
loss_values = EM.train(epochs,visual=True)

#Save em
filename = 'gauss_em.sav'
pickle.dump(EM,open(filename,'wb'))

#Run DIF with initialization EM
epochs = 10000
batch_size = 20000
initial_T = EM.T
initial_w = SoftmaxWeight(K, 2, [64,64,64], mode = 'NN')
initial_w.f[-1].weight = nn.Parameter(torch.zeros(K, 64))
initial_w.f[-1].bias = nn.Parameter(EM.log_pi)
initial_reference = GeneralizedMultivariateNormalReference(2, initial_log_r = torch.log(2.*torch.ones(2)))
dif = DIFDensityEstimator(target_samples,K, initial_T= initial_T, initial_w = initial_w, initial_reference = initial_reference)
loss_values = dif.train(epochs,batch_size,visual=True)

#Save dif
filename = 'gauss_dif.sav'
pickle.dump(dif,open(filename,'wb'))