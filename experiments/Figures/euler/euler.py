import numpy as np
import torch
from matplotlib import image
from torch import nn

torch.manual_seed(0)
number_runs = 10

from models_dif import DIFDensityEstimator, LocationScaleFlow, SoftmaxWeight
from models_em import EMDensityEstimator

rgb = image.imread("euler.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))

for i in range(10,10+number_runs):
    #Sample data according to image
    vector_density = grey.flatten()
    vector_density = vector_density/torch.sum(vector_density)
    lines, columns = grey.shape

    num_samples = 300000
    cat = torch.distributions.Categorical(probs = vector_density)
    categorical_samples = cat.sample([num_samples])
    target_samples = torch.cat([((categorical_samples // columns + torch.rand(num_samples)) / lines).unsqueeze(-1),((categorical_samples % columns + torch.rand(num_samples)) / columns).unsqueeze(-1)],dim=-1)

    #Run EM
    linspace_x = 7
    linspace_y = 7
    K = linspace_x * linspace_y
    EM = EMDensityEstimator(target_samples,K)
    EM.m = torch.cartesian_prod(torch.linspace(0, 1,linspace_x),torch.linspace(0, 1, linspace_y))
    EM.train(200)

    #Run DIF
    initial_T = LocationScaleFlow(K,2)
    initial_T.m = nn.Parameter(EM.m)
    initial_T.log_s = nn.Parameter(EM.log_s)

    initial_w = SoftmaxWeight(K, 2, [128,128,128])
    initial_w.f[-1].bias = nn.Parameter(EM.log_pi)
    initial_w.f[-1].weight = nn.Parameter(torch.zeros(K,initial_w.network_dimensions[-2]))

    dif = DIFDensityEstimator(target_samples,K)
    dif.T = initial_T
    dif.w = initial_w

    epochs = 1000
    batch_size = 30000
    dif.train(epochs, batch_size)

    filename = 'runs/euler_dif' + str(i) + '.sav'
    torch.save(dif,filename)