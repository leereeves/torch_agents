import math
import numpy as np
import random
import torch

class StrategyInterface(object):
    def __init__(self):
        return

    def action_without_prediction(self, action_count, episode, step):
        raise NotImplementedError

    def action_with_prediction(self, action_count, episode, step, prediction):
        raise NotImplementedError

class EpsilonGreedyStrategy(StrategyInterface):
    def __init__(self, epsilon, eval_epsilon):
        super().__init__()
        self.prev_epsilon = None
        self.epsilon = epsilon
        self.eval_epsilon = eval_epsilon

    def __str__(self):
        return f"ε={self.prev_epsilon:0.2f}"

    def action_without_prediction(self, action_count, is_eval):
        if is_eval:
            self.prev_epsilon = float(self.eval_epsilon)
        else:
            self.prev_epsilon = float(self.epsilon)

        if random.random() < self.prev_epsilon:
            action = np.random.randint(action_count)
        else:
            action = None # indicates caller must call action_with_prediction

        return action

    def action_with_prediction(self, action_count, is_eval, prediction):
        action = torch.argmax(prediction).item()
        return action

