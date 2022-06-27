import numpy as np
import torch
from torch.nn import Linear, ReLU

# Fully connected network 
class MLP(torch.nn.Module):
    def __init__(self, layer_sizes, activation=torch.nn.ReLU):
        super().__init__()

        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            if i < len(layer_sizes) - 1:
                layers.append(activation())

        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)
   
# Convolutional network for Atari games, as described in Mnih 2015
class Mnih2015Atari(torch.nn.Module):
    def __init__(self, action_space_size):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size = 8, stride = 4, dtype=torch.float32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, dtype=torch.float32)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, dtype=torch.float32)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512, dtype=torch.float32)
        self.fc5 = torch.nn.Linear(512, action_space_size, dtype=torch.float32)

        self.init_weights()

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        # Without flattening the input tensor the following line gives the error:
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (28672x7 and 3136x512)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)

class ContinuousActorNetwork(torch.nn.Module):
    def __init__(self, state_size, num_actions, hidden1=400, hidden2=300, init_w=3e-3, activation=torch.nn.ReLU):
        super().__init__()
        self.mlp = MLP([state_size, hidden1, hidden2, num_actions], activation=torch.nn.ReLU)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        return self.tanh(self.mlp(x))

class StateActionCriticNetwork(torch.nn.Module):
    def __init__(self, state_size, num_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.mlp = MLP([state_size+num_actions, hidden1, hidden2, 1])

    def forward(self, state, action):
        return self.mlp(torch.cat([state,action],1))


class CategoricalDistFromLogitsNet(torch.nn.Module):
    
    def __init__(self, logits_net):
        super().__init__()
        self.logits_net = logits_net

    def forward(self, state, action=None):
        pi = torch.distributions.categorical.Categorical(logits=self.logits_net(state))
        if action is None:
            action = pi.sample()
        logp = pi.log_prob(action)
        entropy = pi.entropy()
        return action, logp, entropy


class NormalDistFromMeanNet(torch.nn.Module):

    def __init__(self, mean_net, action_size):
        super().__init__()
        self.mean_net = mean_net
        log_std = -0.5 * np.ones(action_size, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, state, action=None):
        mu = self.mean_net(state)
        sigma = torch.exp(self.log_std)
        pi = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        if action is None:
            action = pi.sample()
        # For multidimensional actions, sum the log probabilities
        # to compute the joint log probability
        logp = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()
        return action, logp, entropy

class SqueezeNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.squeeze(self.net(x), -1)

