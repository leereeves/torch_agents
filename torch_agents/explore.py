import math
import numpy as np
import random
import torch

class StrategyInterface(object):
    def __init__(self):
        pass

    def action_without_prediction(self, action_count, episode, step):
        raise NotImplementedError

    def action_with_prediction(self, action_count, episode, step, prediction):
        raise NotImplementedError

class EpsilonGreedyStrategy(StrategyInterface):
    def __init__(self, epsilon, eval_epsilon, min_repeat=1, max_repeat=1):
        super().__init__()
        self.prev_epsilon = None
        self.epsilon = epsilon
        self.eval_epsilon = eval_epsilon
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.sequence = []

    def __str__(self):
        return f"ε={self.prev_epsilon:0.2f}"

    def action_without_prediction(self, action_count, is_eval):
        # Always read epsilon, to advance dynamic hyperparameters
        self.prev_epsilon = float(self.epsilon)
        # Read eval epsilon only in eval epsiodes
        if is_eval:
            self.prev_epsilon = float(self.eval_epsilon)

        # continue previous action sequence, if any
        if len(self.sequence) > 0:
            action = self.sequence.pop()
        # or start a new random action (chance is self.prev_epsilon)
        elif random.random() < self.prev_epsilon:
            low  = int(self.min_repeat)
            high = int(self.max_repeat)
            if low >= high:
                length = 1
            else:
                length = np.random.randint(low, high)
            self.sequence = [np.random.randint(action_count)] * length
            action = self.sequence.pop()
        # or pick a greedy action
        else:
            action = None # indicates caller must call action_with_prediction

        return action

    def action_with_prediction(self, action_count, is_eval, prediction):
        action = torch.argmax(prediction).item()
        return action

class OrnsteinUhlenbeckProcess(object):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1):
        self.mu = mu
        self.sigma = sigma
        self.current_sigma = 0
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def __str__(self):
        return f"σ={self.current_sigma:03f}"

    def sample(self):
        self.current_sigma = float(self.sigma)
        self.prev_x = self.prev_x + self.theta * (self.mu - self.prev_x) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        return self.prev_x

    def reset_states(self):
        self.prev_x = self.x0 if self.x0 is not None else np.zeros(self.size)

class GaussianNoise(object):
    def __init__(self, mu=0.0, sigma=1.0, size=1):
        self.mu = mu
        self.sigma = sigma
        self.current_sigma = 0
        self.size = size

    def __str__(self):
        return f"σ={self.current_sigma:03f}"

    def sample(self):
        self.current_sigma = float(self.sigma)
        return np.random.normal(size=self.size, loc=self.mu, scale=self.current_sigma)

