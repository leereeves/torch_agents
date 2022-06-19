import torch
from torch.nn import Linear, ReLU

# Fully connected network 
class FCNetwork(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            if i < len(layer_sizes) - 1:
                layers.append(torch.nn.ReLU())

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

class ActorNetwork(torch.nn.Module):
    def __init__(self, state_size, num_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, num_actions)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class CriticNetwork(torch.nn.Module):
    def __init__(self, state_size, num_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_size+num_actions, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, 1)
        self.relu = torch.nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        out = self.fc1(torch.cat([state,action],1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
