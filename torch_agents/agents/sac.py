import datetime
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter 

from .. import memory, networks
from ..environments import EnvInterface

from .agent import *

class OffPolicyAgent(Agent):
    def __init__(self, device, env:EnvInterface):
        super().__init__(device)
        self.env = env

    # Convert a Torch tensor to a numpy array,
    # And if that tensor is on the GPU, move to CPU
    def to_numpy(self, tensors):
        return tensors.cpu().numpy()

    # Convert a single object understood by np.asarray to a tensor
    # and move the tensor to our device
    def to_tensor(self, x, dtype = np.float32):
        return torch.tensor(np.asarray(x, dtype=dtype)).to(self.device)


################################################################################
# Soft Actor Critic (SAC) Agent

class SAC(OffPolicyAgent):

    # Hyperparameters to configure the agent, 
    # these may be ints, floats, or schedules 
    # derived from class Schedule
    class Hyperparams(object):
        def __init__(self):
            self.max_actions = 1000
            self.actor_lr = None
            self.critic_lr = None
            self.memory_size = 1e6
            self.minibatch_size = 32
            self.warmup_actions = 0
            self.update_freq = 1
            self.target_update_freq = 1
            self.target_update_rate = 1
            self.clip_rewards = False
            self.gamma = 0.99
            self.alpha = 0.2

    # Network modules required by the agent
    class Modules(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = None
            self.critic1 = None
            self.critic2 = None
            self.q_optimizer = None
            self.actor_optimizer = None

    # Current status of the agent, updated every step
    class Status(object):
        def __init__(self):
            self.action_count = 0
            self.episode_count = 0
            self.update_count = 0
            self.score_history = []

    #######################################################
    # The agent itself begins here
    def __init__(self, env:EnvInterface, hp:Hyperparams, modules:Modules=None, device=None):
        super().__init__(device, env)
        self.hp = hp
        self.status = SAC.Status()
        self.memory = memory.ReplayMemory(self.hp.memory_size)

        state_size = np.array(env.env.observation_space.shape).prod()
        action_size = np.array(env.env.action_space.shape).prod()

        if modules is None:
            modules = SAC.Modules()
            state_size = np.array(env.env.observation_space.shape).prod()
            action_size = np.array(env.env.action_space.shape).prod()
            lows = env.env.action_space.low
            highs = env.env.action_space.high
            mlp = networks.SplitMLP([state_size, 64, 64, action_size], splits=2)
            gaussian = networks.NormalActorFromMeanAndStd(mlp)
            modules.actor = networks.BoundActor(gaussian, mins=lows, maxs=highs)
            modules.critic1 = networks.QMLP(state_size, action_size, [64, 64])
            modules.critic2 = networks.QMLP(state_size, action_size, [64, 64])

        modules.critic1_target = deepcopy(modules.critic1)
        modules.critic2_target = deepcopy(modules.critic2)

        modules.actor = modules.actor.to(self.device)
        modules.critic1 = modules.critic1.to(self.device)
        modules.critic2 = modules.critic2.to(self.device)
        modules.critic1_target = modules.critic1_target.to(self.device)
        modules.critic2_target = modules.critic2_target.to(self.device)
        critic_params = list(modules.critic1.parameters()) + list(modules.critic2.parameters())
        modules.q_optimizer = optim.Adam(critic_params, lr=hp.critic_lr)
        modules.actor_optimizer = optim.Adam(list(modules.actor.parameters()), lr=hp.actor_lr)

        self.modules = modules


    def update_model(self):

        # Many agents don't start training until the replay
        # buffer has enough samples to reduce correlation 
        # in each minibatch of the training data.
        if self.status.action_count < self.hp.warmup_actions:
            return

        # Ensure we have enough memory to sample a full minibatch
        if len(self.memory) < self.hp.minibatch_size:
            return

        # TODO: Update learning rates, which can be dynamic parameters
        #if self.hp.lr is not None:
            #for g in self.actor_opt.param_groups:
            #    g['lr'] = float(self.hp.lr)

        # Update target networks, possibly on a different 
        # schedule than minibatch updates
        if self.status.action_count % self.hp.target_update_freq == 0:
            self.update_targets()

        #####
        # Beyond this point, we begin the parameter update. The code
        # here handles prep steps shared by all agents before
        # calling the agent specific function minibatch_update()

        # Some agents don't update every action
        if self.status.action_count % self.hp.update_freq != 0:
            return

        # All requirements are satisfied, it's time for an update
        self.status.update_count += 1

        # Get a sample from the replay memory
        _, batch, _ = self.memory.sample(self.hp.minibatch_size)
        # Rearrange from list of transitions to lists of states, actions, etc
        list_of_data_lists = list(zip(*batch))
        # Convert to tensors
        states, actions, next_states, rewards, dones = \
            map(self.to_tensor, list_of_data_lists)

        # Update the gradient (every algorithm implements this differently)
        self.minibatch_update(states, actions, next_states, rewards, dones)

    def update_target(self, live, target):
        tau = self.hp.target_update_rate
        for param, target_param in zip(live.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_targets(self):
        self.update_target(self.modules.critic1, self.modules.critic1_target)
        self.update_target(self.modules.critic2, self.modules.critic2_target)

    # Compute the loss and take one step in SGD
    def minibatch_update(self, states, actions, next_states, rewards, dones):
        with torch.no_grad():
            # We compute an unbiased estimate of the Q values of the next 
            # states by using an action sampled from the current policy:
            sampled_actions, sampled_actions_log_p = \
                self.modules.actor(next_states)

            # For stability, we estimate Q values with delayed (target) networks, 
            # and take the minimum of the Q values predicted by two target networks. 
            next_q1 = self.modules.critic1_target(next_states, sampled_actions).squeeze(-1)
            next_q2 = self.modules.critic2_target(next_states, sampled_actions).squeeze(-1)
            next_q = torch.min(next_q1, next_q2)

            # To compute the soft Q value, which maximizes entropy, we add an unbiased 
            # estimate of the entropy, again computed by sampling:
            next_soft_q = next_q - self.hp.alpha * sampled_actions_log_p.squeeze(-1)

            # Then we force the estimated soft Q value of terminal states to be zero:
            next_soft_q_zeroed = next_soft_q * (1 - dones)

            # Finally, the target Q value is computed with the Bellman equation:
            target_q = rewards + self.hp.gamma * next_soft_q_zeroed

        # Compute and minimize the critic loss
        # We train both Q networks to predict the target Q value
        q1 = self.modules.critic1(states, actions).squeeze(-1)
        q2 = self.modules.critic2(states, actions).squeeze(-1)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.modules.q_optimizer.zero_grad()
        q_loss.backward()
        self.modules.q_optimizer.step()

        # Compute and minimize the actor loss
        # Here we maximize the soft Q value
        # (which is entropy, E[-log pi], plus the Q value)
        # by minimizing -1 times the soft Q value
        actions, actions_log_p = self.modules.actor(states)
        q1 = self.modules.critic1(states, actions).squeeze(-1)
        q2 = self.modules.critic2(states, actions).squeeze(-1)
        min_q = torch.min(q1, q2)
        actor_loss = ((self.hp.alpha * actions_log_p) - min_q).mean()

        self.modules.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.modules.actor_optimizer.step()

        # TODO: autotune alpha

    def choose_action(self, state):
        if self.status.action_count < self.hp.warmup_actions:
            # TODO: match action scale to the environment
            action = np.random.rand(1) * 2 - 1
        else:
            with torch.no_grad():
                state_tensor = self.to_tensor(state)
                action, _ = self.modules.actor(state_tensor)
                action = self.to_numpy(action)
        return action

    # Training loop for off-policy agents
    def train(self):

        # Open Tensorboard log
        #path = "./tensorboard/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #tb_log = SummaryWriter(path)

        # Log hyperparameters
        #for key, value in self.config.items():
        #    tb_log.add_text(key, str(value))

        start = time.time()

        self.status.score_history = []
        self.status.episode_count = 0

        state = self.env.reset()
        score = 0
        done = 0
        # This loops through actions (4 frames for Atari, 1 frame for most envs)
        for self.status.action_count in range(int(self.hp.max_actions)):
            action = self.choose_action(state)
            new_state, reward, done, info, life_lost = self.env.step(action)
            self.status.action_count += 1
            score += reward
            if self.hp.clip_rewards:
                clipped_reward = np.sign(reward)
            else:
                clipped_reward = reward
            self.memory.store_transition(state, action, new_state, clipped_reward, done or life_lost)
            self.update_model()
            state = new_state

            if done:
                self.status.episode_count += 1
                elapsed_time = math.ceil(time.time() - start)
                self.status.score_history.append(score)
                moving_average = np.average(self.status.score_history[-20:])

                #tb_log.add_scalars(self.name, {'score': scores[-1]}, self.episode)

                print("Time {}. Episode {}. Action {}. Score {:0.0f}. MAvg={:0.1f}.".format(
                    datetime.timedelta(seconds=elapsed_time), 
                    self.status.episode_count, 
                    self.status.action_count, 
                    score, 
                    moving_average
                    ))

                state = self.env.reset()
                score = 0
                done = 0
            # end if done

            #if self.episode > 0 and self.episode % 10 == 0:
            #    print("Saving model {}".format(self.get_model_filename()))
            #    self.save_model()

        self.env.close()
        #tb_log.close()


"""
References

Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep 
reinforcement learning with a stochastic actor." International conference on 
machine learning. PMLR, 2018.

"""
