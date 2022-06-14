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

