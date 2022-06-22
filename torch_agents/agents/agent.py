import torch

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def hard_update(target, source):
    for t,s  in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)

class Agent(object):
    def __init__(self, device):
        if device is not None:
            self.device = device
        else:        
            if torch.cuda.is_available():
                print("Training on GPU.")
                self.device = torch.device('cuda:0')
            else:
                print("No CUDA device found, or CUDA is not installed. Training on CPU.")
                self.device = torch.device('cpu')

