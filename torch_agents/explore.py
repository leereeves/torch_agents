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
    def __init__(self, 
            random_explore_actions,
            initial_epsilon,
            final_epsilon,
            linear_epsilon_delta,
            eval_epsilon,
            ):
        super().__init__()
        self.epsilon = 1
        self.next_epsilon = initial_epsilon

        self.random_explore_remaining = random_explore_actions
        self.final_epsilon = final_epsilon
        self.delta = linear_epsilon_delta

        self.eval_epsilon = eval_epsilon

    def __str__(self):
        return f"ε={self.epsilon:0.2f}"

    def action_without_prediction(self, action_count, is_eval):
        # a uniform random policy is run for this number of frames
        if self.random_explore_remaining > 0:
            self.random_explore_remaining -= 1
            self.epsilon = 1
        elif is_eval:
            self.epsilon = self.eval_epsilon
        else:
            self.epsilon = self.next_epsilon
            self.next_epsilon = max(self.final_epsilon, self.epsilon - self.delta)

        if random.random() < self.epsilon:
            action = np.random.randint(action_count)
        else:
            action = None # indicates caller must call action_with_prediction

        return action

    def action_with_prediction(self, action_count, is_eval, prediction):
        action = torch.argmax(prediction).item()
        return action

