import torch
from torch import nn
import torch.nn.functional as F

class SoftmaxWeight(nn.Module):
    def __init__(self, K, p, hidden_dimensions =[]):
        super().__init__()
        self.K = K
        self.p = p
        self.network_dimensions = [self.p] + hidden_dimensions + [self.K]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1),nn.Tanh(),])
        network.pop()
        self.f = nn.Sequential(*network)

    def log_prob(self, z):
        unormalized_log_w = self.f.forward(z)
        return unormalized_log_w - torch.logsumexp(unormalized_log_w, dim=-1, keepdim=True)

class ConvNetWeight(nn.Module):
    def __init__(self, K, p):
        super(ConvNetWeight, self).__init__()
        self.K = K
        self.p = p
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, K)

        # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def log_prob(self, z):
        old_shape = list(z.shape)
        if len(old_shape) == 3:
            shape = [old_shape[0]*old_shape[1]]
        elif len(old_shape) == 2:
            shape = [old_shape[0]]
        shape.append(1)
        shape.append(28)
        shape.append(28)
        print(shape)
        old_shape[-1] = self.K
        unormalized_log_w = self.forward(z.view(shape)).reshape(old_shape)
        return unormalized_log_w - torch.logsumexp(unormalized_log_w, dim=-1, keepdim=True)