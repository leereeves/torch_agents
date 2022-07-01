import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter 

from .. import memory, networks, schedule
from ..environments import EnvInterface

from .agent import *

################################################################################
# Soft Actor Critic (SAC) Agent

class SAC(OffPolicyAgent):
    """
    Soft Actor Critic is:

        * a policy gradient algorithm,

        * that attempts to maximize Q-values,

        * and also maximize entropy in the policy distribution,

        * balanced by a temperature hyperparameter.

        * It is off-policy,

        * and model-free.

    Implementation details:

    Inspired by TD3, version 2 of the SAC paper added twin delayed critic 
    networks, which this implementation includes.

    This implementation's default critic for continuous environments 
    uses a Gaussian distribution. This is simple and computationally 
    efficient, but unable to learn multi-modal distributions 
    (distributions with more than one peak). This implementation should,
    however, be compatible with custom critics that use any distribution
    with differentiable log probabilities.
    """

    # Hyperparameters to configure the agent, 
    # these may be ints, floats, or schedules 
    # derived from class Schedule
    class Hyperparams(object):
        def __init__(self):
            self.max_actions = 1000
            "How many actions to take during training, unless stopped early"
            self.actor_lr = None
            "Learning rate for the actor, can be a constant or a schedule"
            self.critic_lr = None
            "Learning rate for the critic, can be a constant or a schedule"
            self.memory_size = 1e6
            "Size of the replay buffer"
            self.minibatch_size = 32
            "Number of transitions in each minibatch"
            self.warmup_actions = 0
            "Random actions to take before training the networks"
            self.update_freq = 1
            "How often to perform a minibatch_update()"
            self.target_update_freq = 1
            "Actions to take between target network updates"
            self.target_update_rate = 1
            "Sometimes called tau, how much to update the target network"
            self.clip_rewards = False
            "If true, all rewards are set to -1, 0, or 1"
            self.gamma = 0.99
            "Discount factor for Q-values"
            self.temperature = 0.2
            "Balances Q-values with entropy (higher favors more entropy)"

    # Network modules required by the agent
    class Modules(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = None
            """
            The actor is a network object derived from torch.nn.Module that accepts a 
            state and optional action, predicts a policy (a probability distribution
            over actions) from the state, samples an action if None was provided,
            and returns the action and the log probability of that action. 

            If no actor is provided, the default actor for environments with discrete 
            action spaces is an MLP that predicts a Categorical distribution.
            The default actor for continuous action spaces is an MLP that predicts
            the mean and standard deviation of a Gaussian distribution which is then
            bound to (-1, 1) by Tanh and finally scaled to match the environment.
            """
            self.critic1 = None
            self.critic2 = None
            """
            The critics are network objects derived from torch.nn.Module that accept
            a state and action and predict a single soft Q value.

            If no critics are provided, the default critics are MLPs that concatenate
            the state and action before the first layer.
            """
            self.actor_optimizer = None
            """
            An optimization algorithm from (or compatible with) torch.optim 
            for the actor parameters. If none is provided, Adam will be used.
            """
            self.critic_optimizer = None
            """
            An optimization algorithm from (or compatible with) torch.optim 
            for the parameters of both critic networks. If none is provided, 
            Adam will be used.
            """

    # Current status of the agent, updated every step
    class Status(object):
        def __init__(self):
            self.action_count = 0
            "How many actions the agent has taken"
            self.update_count = 0
            "How many minibatch updates the agent has performed"
            self.episode_count = 0
            "How many episodes the agent has completed"
            self.score_history = []
            "All scores from completed episodes in chronological order"
            self.elapsed_time = 0
            "Elapsed time since the start of training"

    #######################################################
    # The agent itself begins here
    def __init__(self, env:EnvInterface, hp:Hyperparams, modules:Modules=None, device=None):
        super().__init__(device, env)
        self.hp = deepcopy(hp)
        self.current = deepcopy(hp)
        self.update_hyperparams()

        self.status = SAC.Status()
        self.memory = memory.ReplayMemory(self.current.memory_size)

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
        modules.critic_optimizer = optim.Adam(critic_params, lr=self.current.critic_lr)
        modules.actor_optimizer = optim.Adam(list(modules.actor.parameters()), lr=self.current.actor_lr)

        self.modules = modules

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
            next_soft_q = next_q - self.current.temperature * sampled_actions_log_p.squeeze(-1)

            # Then we force the estimated soft Q value of terminal states to be zero:
            next_soft_q_zeroed = next_soft_q * (1 - dones)

            # Finally, the target Q value is computed with the Bellman equation:
            target_q = rewards + self.current.gamma * next_soft_q_zeroed

        # Compute and minimize the critic loss
        # We train both Q networks to predict the target Q value
        q1 = self.modules.critic1(states, actions).squeeze(-1)
        q2 = self.modules.critic2(states, actions).squeeze(-1)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.modules.critic_optimizer.zero_grad()
        q_loss.backward()
        self.modules.critic_optimizer.step()

        # Compute and minimize the actor loss
        # Here we maximize the soft Q value
        # (which is entropy, E[-log pi], plus the Q value)
        # by minimizing -1 times the soft Q value
        actions, actions_log_p = self.modules.actor(states)
        q1 = self.modules.critic1(states, actions).squeeze(-1)
        q2 = self.modules.critic2(states, actions).squeeze(-1)
        min_q = torch.min(q1, q2)
        actor_loss = ((self.current.temperature * actions_log_p) - min_q).mean()

        self.modules.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.modules.actor_optimizer.step()

        # TODO: autotune temperature

    def choose_action(self, state):
        if self.status.action_count < self.current.warmup_actions:
            # TODO: handle multidimensional actions, scale and bound actions
            action = np.random.rand(1) * 2 - 1
        else:
            with torch.no_grad():
                state_tensor = self.to_tensor(state)
                action, _ = self.modules.actor(state_tensor)
                action = self.to_numpy(action)
        return action

    def on_episode_end(self):
        #tb_log.add_scalars(self.name, {'score': scores[-1]}, self.episode)

        moving_average = np.average(self.status.score_history[-20:])

        print("Time {}. Episode {}. Action {}. Score {:0.0f}. MAvg={:0.1f}. lr={:0.2e} {:0.2e} temp={:0.3f}".format(
            datetime.timedelta(seconds=self.status.elapsed_time), 
            self.status.episode_count, 
            self.status.action_count, 
            self.status.score_history[-1], 
            moving_average,
            self.current.actor_lr,
            self.current.critic_lr,
            self.current.temperature
            ))



"""
References

Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep 
reinforcement learning with a stochastic actor." International conference on 
machine learning. PMLR, 2018.

"""
