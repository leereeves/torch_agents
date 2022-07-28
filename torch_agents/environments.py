# Wraps reinforcement learning tasks provided by OpenAI Gym

import gym
import numpy as np
import random

class EnvInterface(object):
    def __init__(self, name):
        self.name = name
        return

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class GymEnv(EnvInterface):
    def __init__(self, name, render_mode=None, valid_actions=None, action_repeat=1, frameskip=None):
        super().__init__(name)
        self.render_mode = render_mode
        self.actions = valid_actions
        self.action_repeat = action_repeat
        if frameskip is None:
            self.env = gym.make(name)
        else:
            self.env = gym.make(name, frameskip=frameskip)

    def seed(self, seed):
        return self.env.seed(seed)

    def action_space_size(self):
        return len(self.actions)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # The list self.actions is a map from action_indexes to action numbers
        # if the valid actions are discrete and aren't [0, ..., n]
        if self.actions is not None:
            action = self.actions[action]

        # Some environments expect action as a Discrete, which is just a scalar
        # others expect a Box(), which is an array of continuous values
        # if not isinstance(self.env.action_space, gym.spaces.Discrete):
        #     action = np.array([action])

        if self.render_mode == 'human':
            self.env.render()

        return self.env.step(action) + (False, )
        
    def render(self):
        raise NotImplementedError # use render_mode = 'human' instead

    def close(self):
        return self.env.close()


class GymAtariEnv(GymEnv):
    def __init__(self, name, render_mode=None, valid_actions=None, action_repeat=1):
        super().__init__(name, render_mode, valid_actions, action_repeat, frameskip=1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, scale_obs=True)
        self.env = gym.wrappers.FrameStack(self.env, num_stack=action_repeat, lz4_compress=True)

    def reset(self):
        self.current_lives = None
        return self.env.reset()

    # Overridden and hidden because Atari games 
    # in OpenAI gym don't support this method any more
    def render(self):
        return 

    def step(self, action_index):
        state, reward, done, info, life_lost = super().step(action_index)
        lives = info['lives']
        life_lost = (self.current_lives is not None and lives != self.current_lives)
        self.current_lives = lives
        return state, reward, done, info, life_lost
