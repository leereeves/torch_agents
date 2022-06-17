# Two implementations of replay memory
#
# class ReplayMemory is a simple flat array from which
# past transistions are uniformly sampled.
#
# class PrioritizedReplayMemory returns transitions with large
# temporal-difference error (|target-prediction|)more frequently,
# inspired by Schaul (2015) https://arxiv.org/abs/1511.05952v4
# but using periodic systematic resets rather than randomization
# to schedule review of past transitions whose target values
# may have changed.

import math
import random

from . import sumtree

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = [None] * self.capacity
        self.allocated = 0
        self.index = 0

    def __len__(self):
        return self.allocated

    def store_transition(self, state, action, new_state, reward, done):
        self.buffer[self.index] = (state, action, new_state, reward, done)
        if (self.allocated + 1) < self.capacity:
            self.allocated += 1
            self.index += 1
        else:
            self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indexes = random.sample(range(self.allocated), k=batch_size)
        samples = [self.buffer[i] for i in indexes]
        weights = [1] * batch_size
        return indexes, samples, weights

    def update_weight(self, index, weight):
        return

class PrioritizedReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = sumtree.SumTree(self.capacity)

    def __len__(self):
        return self.tree.allocated

    def store_transition(self, state, action, new_state, reward, done):
        t = (state, action, new_state, reward, done)
        self.tree.push(t, 1)

    # Draw a random sample of batch_size transitions by priority
    # (that is, the probability of drawing a transition
    # is proportional to the priority of that transition)
    def sample(self, batch_size):
        indexes = []
        samples = []
        weights = []

        N = len(self)
        while len(indexes) < batch_size:
            r = random.random() * self.tree.get_total_weight()
            i = self.tree.get_index_by_weight(r)
            
            if i in indexes:
                continue

            weight = self.tree.get_weight(i)
            total_weight = self.tree.get_total_weight()
            # to avoid division by zero, replace with smallest positive float
            if weight == 0:
                weight = math.getnextafter(0.0, math.inf)
            if total_weight == 0:
                total_weight = math.getnextafter(0.0, math.inf)
            # calculate the weight for importance sampling (see Schaul 2016)
            p = weight / total_weight
            isw = (1/N) * (1/p)
            # Set the weight of the selected transition to zero
            # to reduce the chance it will be drawn again until the agent 
            # updates its priority (that should happen soon after sampling).
            self.tree.set_weight(i, 0)
            indexes.append(i)
            samples.append(self.tree.get_data(i))
            weights.append(isw)

        return indexes, samples, weights

    def update_weight(self, index, weight):
        self.tree.set_weight(index, weight)
        return
