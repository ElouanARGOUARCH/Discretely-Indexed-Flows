import torch
from models_dif import SoftmaxWeight, DIFDensityEstimator

###MNIST###

import torchvision.datasets as datasets
import matplotlib.pyplot as plt
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
images = mnist_trainset.data.flatten(start_dim=1)
targets = mnist_trainset.targets

digit = 'all'
if digit != 'all':
    extracted = images[targets == digit].float()
else:
    extracted = images.float()
target_samples = extracted

num_samples = target_samples.shape[0]
print('number of samples = ' + str(num_samples))
p = target_samples.shape[-1]
plt.imshow(target_samples[torch.randint(low = 0, high = num_samples, size = [1])].reshape(28,28))

train_set, test_set = target_samples[:4000], target_samples[4000:]

K = 100
dif = DIFDensityEstimator(target_samples, K)
dif.w = SoftmaxWeight(K,p, [256,256,256])
dif.train(1000, 6000)

filename = 'dif_mnist.sav'
torch.save(dif, filename)