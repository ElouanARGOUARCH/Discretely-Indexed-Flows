import torch
from models_dif import SoftmaxWeight, DIFDensityEstimator, LocationScaleFlow

###MNIST###

import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
images = mnist_trainset.data.flatten(start_dim=1).float()
temp = (images + torch.rand_like(images))/256

def pre_process(x, lbda):
    return torch.logit(lbda*torch.ones_like(x) + x*(1-2*lbda))

def inverse_pre_process(x, lbda):
    return torch.sigmoid((x- lbda*torch.ones_like(x))/(1-2*lbda))

lbda = 1e-6
target_samples = pre_process(temp, lbda)
p = target_samples.shape[-1]

K = 5
model = DIFDensityEstimator(target_samples, K)
model.reference.cov += 1e-3*torch.eye(p)
initial_w = SoftmaxWeight(K, p, [512,512,256,256,128,128])
initial_w.f[-1].bias = torch.nn.Parameter(torch.ones(K)/K)
initial_w.f[-1].weight = torch.nn.Parameter(torch.zeros(K,initial_w.network_dimensions[-2]))
initial_T = LocationScaleFlow(K,p)
initial_T.m = torch.nn.Parameter(torch.zeros(K,p) + 0.1*torch.randn(K,p))
initial_T.log_s = torch.nn.Parameter(torch.zeros(K,p))
model.w = initial_w
model.T = initial_T
model.train(1000,6000)
filename = 'dif_mnist_second_model.sav'
torch.save(model, filename)
