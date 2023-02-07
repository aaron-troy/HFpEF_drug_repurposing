import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):

    # Constructor for a single layer AE with leaky Relu non-linearity
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()

        self.net = nn.Sequential(nn.Linear(input_size, latent_size, bias=False),
                                 Nonlinearity(),
                                 nn.Linear(latent_size, input_size, bias=False))

    def forward(self, x):
        return self.net(x)
    
    def set_weights(self, weights):
        
        weights_encode = torch.nn.Parameter(weights['state_dict']['net.0.weight'])
        weights_decode = torch.nn.Parameter(weights['state_dict']['net.2.weight'])
        
        self.net[0].weight = weights_encode
        self.net[2].weight = weights_decode

class Nonlinearity(torch.nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        # return F.selu(x)
        # return F.relu(x)
        # return x
        #return F.tanh(x)
        return F.leaky_relu(x)
        # return x + torch.sin(10*x)/5
        # return x + torch.sin(x)
        # return x + torch.sin(x) / 2
        # return x + torch.sin(4*x) / 2
        # return torch.cos(x) - x
        # return x * F.sigmoid(x)
        # return torch.exp(x)#x**2
        # return x - .1*torch.sin(5*x)

