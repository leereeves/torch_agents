from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# An MLP that accepts a state and action as inputs and returns a single value   
class QMLP(torch.nn.Module):
    def __init__(self, state_size, num_actions, hidden_sizes, activation=torch.nn.ReLU):
        super().__init__()
        layer_sizes = [state_size+num_actions] + hidden_sizes + [1]
        self.mlp = MLP(layer_sizes, activation)

    def forward(self, state, action):
        return self.mlp(torch.cat([state,action],1))


# An MLP that splits before the last layer and has multiple output layers
class SplitMLP(torch.nn.Module):
    def __init__(self, layer_sizes, splits, activation=nn.ReLU):
        super().__init__()
        output_size = layer_sizes.pop(-1)
        self.mlp = MLP(layer_sizes, activation)
        self.outputs = nn.ModuleList([torch.nn.Linear(layer_sizes[-1], output_size) for _ in range(splits)])
        self.f = activation()

    def forward(self, x):
        results = []
        x = self.f(self.mlp(x))
        for o in self.outputs:
            results.append(o(x))
        return tuple(results)


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

class NormalActorFromMeanAndStd(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, state, action=None):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5

        mu, log_sigma = self.net(state)
        # Bound the log standard deviation between LOG_STD_MIN and LOG_STD_MAX
        # The range -5 to 2 comes from CleanRL, who attribute it to SpinUp / Denis Yarats
        log_sigma = (torch.tanh(log_sigma) + 1) / 2
        log_sigma = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * log_sigma
        sigma = torch.exp(log_sigma)
        policy = torch.distributions.Normal(loc=mu, scale=sigma)
        # If no action is given, take a random sample
        if action is None:
            action = policy.rsample()
        logp = policy.log_prob(action)
        entropy = policy.entropy()
        return action, logp, entropy

# BoundActor implements "Enforcing Action Bounds" from Appendix C of 
# the soft-actor critic paper: Haarnoja, Tuomas, et al., 2018
# An action, log probability pair from an unbounded distribution 
# with infinite support is bounded between mins and maxes, and
# the log probability of this bound action is adjusted appropriately.
class BoundActor(nn.Module):
    def __init__(self, actor, mins, maxs) -> None:
        super().__init__()
        # actor is a nn.Module whose forward function returns action, logp
        self.actor = actor
        # scale is divided by 2 because the range of tanh is 2 (-1 to 1)
        self.scale = nn.Parameter(torch.FloatTensor((maxs - mins) / 2.0), requires_grad=False)
        # bias is just the midpoint of min and max
        self.bias = nn.Parameter(torch.FloatTensor((maxs + mins) / 2.0), requires_grad=False)

    def forward(self, state, action=None):
        # Save a tiny constant to add before taking log, to avoid log 0
        tiny_float = torch.finfo(torch.float16).tiny 

        # Get the action and log probability from the unbound actor
        u, logp_u, entropy = self.actor(state, action)

        # Compute the new log probability under the change of variables 
        # a_i = scale_i * tanh(u_i) + bias_i
        tanh_u = torch.tanh(u)
        da_over_du = self.scale * (1 - tanh_u**2)
        log_da_over_du = torch.log(da_over_du + tiny_float)
        logp_a = logp_u - log_da_over_du.sum(-1, keepdim=True)

        # Compute the bound action
        action = tanh_u * self.scale + self.bias

        # Estimate a correction for the entropy under the change of variables. 
        # a_i = scale_i * tanh(u_i) + bias_i
        # Reference:
        # Hnizdo, Vladimir, and Michael K. Gilson. 
        # "Thermodynamic and differential entropy under a change of variables." 
        # Entropy 12.3 (2010): 578-590.
        # https://www.mdpi.com/1099-4300/12/3/578
        # The exact formula for the correction is an expectation across all
        # possible actions, which is intractable in general. Therefore, this
        # entropy is only an estimate and should only be used for diagnostics.
        entropy += log_da_over_du.sum(-1, keepdim=True)

        # Return the bound action, the log probability of that action, 
        # and the total entropy of the policy.
        return action, logp_a, entropy


class Squeeze(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.squeeze(self.net(x), -1)

