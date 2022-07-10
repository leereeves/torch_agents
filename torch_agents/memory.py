import math
import numpy as np
import random
import scipy.signal

from . import sumtree

###############################################################################
# Two implementations of replay memory for off-policy algorithms
#
# class ReplayMemory is a simple flat array from which
# past transistions are uniformly sampled.
#
# class PrioritizedReplayMemory returns transitions with large
# temporal-difference error (|target-prediction|)more frequently,
# See Schaul (2015) https://arxiv.org/abs/1511.05952v4

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

###############################################################################
# Memory for on-policy algorithms that use 
# generalized advantage estimation, like PPO

class OnPolicyAdvantageMemory:
    def __init__(self, capacity, gamma=0.99, lambd=0.95):
        self.capacity = capacity
        self.states = [None] * capacity
        self.actions = [None] * capacity
        self.new_states = [None] * capacity
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values  = np.zeros(capacity, dtype=np.float32)
        self.logps   = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.gamma = gamma
        self.lambd = lambd
        self.reset()

    def reset(self):
        # Reset to beginning
        self.index = 0
        self.episode_start_index = 0

    def store_transition(self, state, action, new_state, reward, done, value, logp):
        assert (self.index < self.capacity)
        self.states[self.index] = state
        self.actions[self.index] = action
        self.new_states[self.index] = new_state
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.values[self.index] = value
        self.logps[self.index] = logp
        self.index += 1


    def discounted_cumulative_sum(self, x, discount):
        """
        This trick for computing discounted cumulative sum using 
        a scipy filter is discussed here:
        "We'd like to calculate C[i] satisfying the recurrence C[i] = R[i] + discount * C[i+1]"
        https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
        From [x_0, ..., x_n]
        Calculate: 
        [
        x_0 + discount * x_1 + ... + discount^n * x_n,
        x_1 + discount * x_2 + ... + discount^{n-1} * x_n,
        ...
        x_n
        ]
        """
        r = x[::-1]
        a = [1, -discount]
        b = [1]
        y = scipy.signal.lfilter(b, a, x=r, axis=0)
        return y[::-1]

    def end_episode(self, last_value):
        # Slice out this episode
        episode_slice = slice(self.episode_start_index, self.index)

        # Mark start of next episode
        self.episode_start_index = self.index

        # Append value of last state (in case the episode terminated early)
        rewards = np.append(self.rewards[episode_slice], last_value)
        values = np.append(self.values[episode_slice], last_value)
        
        # Calculate returns
        self.returns[episode_slice] = self.discounted_cumulative_sum(rewards, self.gamma)[:-1]

        # Calculate advantages (GAE-Lambda), which are the difference 
        # between the result of the Bellman equation and the estimated value
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[episode_slice] = self.discounted_cumulative_sum(deltas, self.gamma * self.lambd)
        
    def get(self):
        assert (self.index == self.capacity)

        # Normalize the advantage
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / adv_std

        return self.states, self.actions, self.new_states, self.rewards, self.dones, \
               self.returns, self.advantages, self.logps

