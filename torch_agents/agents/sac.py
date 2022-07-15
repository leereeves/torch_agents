import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import types

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter 

from .. import memory, networks, schedule
from ..environments import EnvInterface

from .agent import *

################################################################################
# Soft Actor Critic (SAC) Agent for environments with continuous action spaces

class ContinuousSAC(OffPolicyAgent):
    """Implements Soft Actor Critic (Haarnoja, Tuomas, et al, 2018).

    Soft Actor Critic (SAC) is a model-free, off-policy, policy gradient 
    algorithm that maximizes a combination of Q-values plus entropy 
    in the policy distribution. The balance between expected returns
    and entropy is controlled by a temperature hyperparameter.

    This version of SAC supports continuous actions. See SACDiscrete
    for a version of SAC that supports discrete actions.

    **Implementation details**

    Version 2 of the SAC paper added twin delayed (target) critic 
    networks inspired by TD3, which this implementation includes.

    This implementation's default actor for continuous environments 
    creates a policy with a Gaussian distribution. This is simple and
    efficient, but unable to learn multi-modal distributions. This 
    implementation should, however, be compatible with custom actors 
    that create policies with any distribution that has differentiable
    action samples (e.g. via reparameterization).

    **References**

    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy 
    deep reinforcement learning with a stochastic actor." International 
    conference on machine learning. PMLR, 2018.

    http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf

    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." 
    arXiv preprint arXiv:1812.05905 (2018).

    https://arxiv.org/pdf/1812.05905.pdf

    **Hyperparameters and other members are organized in several subclasses:**

    Attributes:
        hp (Hyperparams): Initial values or schedules for hyperparameters
        current (Hyperparams): Current values of hyperparameters
        status (Status): Current values of public status variables
        modules (Modules): Modules and optimizers
    """

    class Hyperparams(object):
        """
        Hyperparameters to configure an SAC agent. These may be ints, floats, or 
        schedules derived from class Schedule, allowing any hyperparameter
        to change dynamically during training.
        """
        def __init__(self):
            self.max_actions = 1000
            """How many actions to take during training, unless stopped early"""
            self.actor_lr = 3e-4
            """Learning rate for the actor optimizer. This can be used both to 
            initialize the default optimizer and to provide adaptive learning 
            rate schedules for any optimizer. Default value is 3e-4."""
            self.critic_lr = 3e-4
            """Learning rate for the critic optimizer. This can be used both to 
            initialize the default optimizer and to provide adaptive learning 
            rate schedules for any optimizer. Default value is 3e-4."""
            self.temperature_lr = 3e-4
            """Learning rate for the temperature autotuning optimizer. This can 
            be used both to initialize the default optimizer and to provide 
            adaptive learning rate schedules for any optimizer. Default value 
            is 3e-4."""
            self.memory_size = 1e6
            "Size of the replay buffer. Default value is 1e6."
            self.minibatch_size = 256
            "Number of transitions in each minibatch. Default value is 256."
            self.warmup_actions = 0
            "Random actions to take before training the networks. Default is zero."
            self.update_freq = 1
            """How many actions to perform between minibatch_update() calls.
            Default value is 1, which performs an update after every action."""
            self.minibatches_per_update = 1
            """How many minibatch gradients to compute and apply during an update.
            Default value is 1."""
            self.target_update_freq = 1
            """How many actions to perform between target network updates.
            (A target network update is when values are copied from the live
            networks to the delayed networks used to estimate target Q values.)
            Default value is 1, which performs an update after every action."""
            self.target_update_rate = 0.005
            r"""Represented by the Greek letter tau (:math:`\tau`) 
            in target update equations. Used for soft target network updates:
            
            .. math:: \theta_{T} = \tau \theta_{L} + (1 - \tau) \theta_{T}
            """
            self.clip_rewards = False
            "If true, all rewards are clipped to the interval [-1, 1]."
            self.reward_scale = 1
            """All rewards are multiplied by this number. Default value is 
            1. 
            
            This is intended for compatibility with early SAC examples;
            new code should use the temperature hyperparamter, whose
            effect is similar (but with inverse values)."""
            self.gamma = 0.99
            r"""Discount factor for Q-values in the temporal difference equation:
            
            .. math:: Q(s_t, a_t) = E_{s_{t+1}}[r_t + \gamma * \max_{a_{t+1}} Q(s_{t+1},a_{t+1})]"""
            self.temperature = 0.2
            """
            Balances Q-values with entropy (higher favors more entropy).
            The effect of temperature varies with the scale and frequency of
            rewards, so it is generally preferable to use target_entropy.
            Despite that, this is the default hyperparameter for entropy 
            maximization because it is easier to understand.
            """
            self.target_entropy = None
            """
            If set, the temperature is tuned automatically to drive the
            average entropy toward target_entropy, and the temperature
            hyperparameter is ignored.
            """
            self.hidden_size = 256
            """Width of hidden layers in the default networks. Ignored
            if custom networks are provided."""

    # Network modules and optimizers required by the agent
    class Modules(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = None
            """
            A network object derived from torch.nn.Module that includes
            a forward function with the following signature:

            forward(state, action = None)

            * state: a tensor of shape (batch size, ) + env.observation_space.shape

            * action: an optional tensor of shape (batch size, ) + env.action_space.shape

            that predicts a policy (a probability distribution over actions) 
            from the state, samples an action if None was provided,
            and returns the action, log probability of that action, and entropy as
            
            return value: (action, logp, entropy)
            
            * action: a tensor of shape (batch size, ) + env.action_space.shape

            * logp: a tensor of shape (batch size, 1)

            * entropy: a tensor of shape (batch size, 1)

            If no actor is provided, the default actor for continuous action spaces 
            is an MLP that predicts the mean and standard deviation of a Gaussian 
            distribution which is then limited to (-1, 1) by Tanh and finally scaled 
            and recentered to match the environments action_range.
            """
            self.critic1 = None
            "First critic network"
            self.critic2 = None
            """Second critic network
            
            The critics are network objects derived from torch.nn.Module that include
            a forward function with the following signature:
            
            forward(state, action)

            * state: a tensor of shape (batch size, ) + env.observation_space.shape

            * action: a tensor of shape (batch size, ) + env.action_space.shape

            return value: q
            
            * q: a single soft Q value of shape (batch size, 1)

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
            "How many updates the agent has performed"
            self.minibatch_count = 0
            "How many minibatch gradients the agent has applied"
            self.episode_count = 0
            "How many episodes the agent has completed"
            self.score_history = []
            "All scores from completed episodes in chronological order"
            self.elapsed_time = 0
            "Elapsed time since the start of training"
            self.entropy = 0
            "Average entropy in the most recent minibatch of policy distributions pi(s_t)"

    #######################################################
    # The agent itself begins here
    def __init__(self, env:EnvInterface, hp:Hyperparams, modules:Modules=None, device=None):
        """Initialize the agent.

        Args:
            env (EnvInterface): An environment from torch_agents.environments
            hp (Hyperparams): Initial values or schedules for hyperparameters
            device: A torch.device, a string or int, or None to autodetect.
        """
        super().__init__(device, env)
        self.hp = deepcopy(hp)
        self.current = deepcopy(hp)
        self._update_hyperparams()

        self.status = ContinuousSAC.Status()
        self._internals = types.SimpleNamespace()
        self._internals.action_space = env.env.action_space

        self.memory = memory.ReplayMemory(self.current.memory_size)

        state_size = np.array(env.env.observation_space.shape).prod()
        action_size = np.array(env.env.action_space.shape).prod()

        if modules is None:
            modules = ContinuousSAC.Modules()
            state_size = np.array(env.env.observation_space.shape).prod()
            action_size = np.array(env.env.action_space.shape).prod()
            lows = env.env.action_space.low
            highs = env.env.action_space.high
            h = int(self.current.hidden_size)
            mlp = networks.SplitMLP([state_size, h, h, action_size], splits=2)
            gaussian = networks.NormalActorFromMeanAndStd(mlp)
            modules.actor = networks.BoundActor(gaussian, mins=lows, maxs=highs)
            modules.critic1 = networks.QMLP(state_size, action_size, [h, h])
            modules.critic2 = networks.QMLP(state_size, action_size, [h, h])

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

        # Optimizer required to automatically adjust temperature
        if self.current.target_entropy is not None:
            self._internals.log_temperature = nn.Parameter(torch.zeros(1).to(self.device))
            modules.entropy_optimizer = optim.Adam([self._internals.log_temperature], lr=self.current.temperature_lr)
            self.hp.temperature = 0
        self.modules = modules

    def train(self):
        super().train()

    def _update_learning_rates(self):
        self._update_lr(self.modules.actor_optimizer, self.current.actor_lr)
        self._update_lr(self.modules.critic_optimizer, self.current.critic_lr)
        if self.current.target_entropy is not None:
            self._update_lr(self.modules.entropy_optimizer, self.current.temperature_lr)

    def _update_targets(self):
        """
        Update target networks from live networks, called automatically by train()
        """
        self._update_target(self.modules.critic1, self.modules.critic1_target)
        self._update_target(self.modules.critic2, self.modules.critic2_target)

    def _minibatch_update(self, states, actions, next_states, rewards, dones):
        """
        Update live networks by gradient descent, called automatically by train()

        Arguments:
        
        * states: a tensor of shape (minibatch size, ) + env.observation_space.shape
        
        * actions: a tensor of shape (minibatch size, ) + env.action_space.shape
        
        * next_states: a tensor of shape (minibatch size, ) + env.observation_space.shape
        
        * rewards: a tensor of shape (minibatch size, )
        
        * dones: a tensor of shape (minibatch size, )
        """
        # Use tuned temperature if we're doing that
        if self.current.target_entropy is not None:
            self.current.temperature = self._internals.log_temperature.exp().item()

        with torch.no_grad():
            # We compute an unbiased estimate of the Q values of the next 
            # states by using an action sampled from the current policy:
            sampled_actions, sampled_actions_log_p, _ = \
                self.modules.actor(next_states)

            # For stability, we estimate Q values with delayed (target) networks, 
            # and take the minimum of the Q values predicted by two target networks. 
            next_q1 = self.modules.critic1_target(next_states, sampled_actions).squeeze(-1)
            next_q2 = self.modules.critic2_target(next_states, sampled_actions).squeeze(-1)
            next_q = torch.min(next_q1, next_q2)

            # To compute the soft Q value, which maximizes entropy, we add an unbiased 
            # estimate of the entropy, again computed by sampling:
            next_soft_q = next_q - self.current.temperature * sampled_actions_log_p.sum(-1)

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
        actions, actions_log_p, actor_entropy = self.modules.actor(states)
        q1 = self.modules.critic1(states, actions).squeeze(-1)
        q2 = self.modules.critic2(states, actions).squeeze(-1)
        min_q = torch.min(q1, q2)
        actor_loss = ((self.current.temperature * actions_log_p.sum(-1)) - min_q).mean()

        self.modules.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.modules.actor_optimizer.step()

        self.status.entropy = actor_entropy.mean()

        # Automatically adjust temperature, see reference 2
        if self.current.target_entropy is not None:
            # Recompute probabilities with new actor parameters
            with torch.no_grad():
                _, log_p, _ = self.modules.actor(states)
            # Optimize a Monte Carlo estimate of the loss function
            # from equation 18 in reference 2:
            # J(\alpha) = E_{a_t \tilde \pi_t} [-\alpha \log \pi_t(a_t | s_t) - \alpha \bar H]
            alpha = self._internals.log_temperature.exp()
            temp_loss = (alpha * (-log_p - self.current.target_entropy)).mean()

            self.modules.entropy_optimizer.zero_grad()
            temp_loss.backward()
            self.modules.entropy_optimizer.step()


    def _choose_action(self, state):
        """
        Choose an action from the given state, called automatically by train()

        * state: a tensor of shape env.observation_space.shape

        return value: action

        * action: a numpy array or a tensor of shape env.action_space.shape
        """
        if self.status.action_count < self.current.warmup_actions:
            # Sample random numbers in uniform(0, 1) with shape to match the action_space
            d = np.array(self._internals.action_space.shape).prod()
            r = np.random.rand(d)
            # Rescale and offset 
            scale = self._internals.action_space.high - self._internals.action_space.low
            action = r * scale + self._internals.action_space.low
            self.status.entropy = np.log(scale).sum()
        else:
            # Ask the actor to choose the action
            with torch.no_grad():
                action, _, _ = self.modules.actor(state)
        return action

    def on_episode_end(self):
        """
        Called automatically by train() at the end of an episode
        """
        #tb_log.add_scalars(self.name, {'score': scores[-1]}, self.episode)

        moving_average = np.average(self.status.score_history[-20:])

        print("Time {}. Episode {}. Action {}. Score {:0.0f}. MAvg={:0.1f}. lr={:0.2e} {:0.2e} temp={:0.3f} entropy={:0.3f}".format(
            datetime.timedelta(seconds=self.status.elapsed_time), 
            self.status.episode_count, 
            self.status.action_count, 
            self.status.score_history[-1], 
            moving_average,
            self.current.actor_lr,
            self.current.critic_lr,
            self.current.temperature,
            self.status.entropy
            ))
