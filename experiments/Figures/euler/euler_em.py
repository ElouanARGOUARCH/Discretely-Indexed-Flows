import numpy as np
import torch
from matplotlib import image
from torch import nn

torch.manual_seed(0)
number_runs = 20

from models_em import FullRankEMDensityEstimator

rgb = image.imread("euler.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))

for i in range(number_runs):
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
    EM = FullRankEMDensityEstimator(target_samples,K)
    EM.m = torch.cartesian_prod(torch.linspace(0, 1,linspace_x),torch.linspace(0, 1, linspace_y))
    EM.train(200)

    filename = 'runs_EM/euler_em' + str(i) + '.sav'
    torch.save(EM,filename)