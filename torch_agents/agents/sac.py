import datetime
import gym
import numpy as np
import os.path
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import types
from pprint import pprint

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter 

from .. import memory, networks, schedule
from ..environments import EnvInterface

from .agent import *

################################################################################
# Soft Actor Critic (SAC) Agent 

class SAC(OffPolicyAgent):
    """Implements Soft Actor Critic (Haarnoja, Tuomas, et al, 2018).

    Soft Actor Critic (SAC) is a model-free, off-policy, policy gradient 
    algorithm that maximizes a combination of Q-values plus entropy 
    in the policy distribution. The balance between expected returns
    and entropy is controlled by a temperature hyperparameter.

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
        model (Model): modules and optimizers
    """

    class Hyperparams(OffPolicyAgent.Hyperparams):
        """
        Hyperparameters to configure an SAC agent. These may be ints, floats, or 
        schedules derived from class Schedule, allowing any hyperparameter
        to change dynamically during training.
        """
        def __init__(self):
            super().__init__()
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
            self.reward_scale = 1
            """All rewards are multiplied by this number. Default value is 
            1. 
            
            This is intended for compatibility with early SAC examples;
            new code should use the temperature hyperparameter, whose
            effect is similar (roughly, temperature = 1/reward_scale)
            except that reward_scale also influences critic_lr."""
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
            self.hidden_depth = 2
            """Number of hidden layers in the default networks. Ignored
            if custom networks are provided."""
            self.hidden_size = 256
            """Width of hidden layers in the default networks. Ignored
            if custom networks are provided."""

    # Network model and optimizers required by the agent
    class Model(nn.Module):
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
    class Status(OffPolicyAgent.Status):
        def __init__(self):
            super().__init__()
            self.entropy = 0
            "Average entropy in the most recent minibatch of policy distributions pi(s_t)"

    #######################################################
    # The agent itself begins here
    def __init__(self, 
                    env:EnvInterface, 
                    hp:Hyperparams=None, 
                    model:Model=None, 
                    device=None,
                    checkpoint=None):
        """Initialize the agent.

        Args:
            env (EnvInterface): An environment from torch_agents.environments
            hp (Hyperparams): Initial values or schedules for hyperparameters
            device: A torch.device, a string or int, or None to autodetect.
        """

        # Create hp and current members to organize hyperparameters
        if checkpoint is not None:
            hp = self._load_checkpoint_first_pass(checkpoint)
        elif hp is None:
            hp = SAC.Hyperparams()
            print("Using the default hyperparameters:")
            pprint(vars(hp))

        super().__init__(device, env, hp)

        # Create replay memory
        self.memory = memory.ReplayMemory(self.current.memory_size)

        # Create Status object to organize public status variables
        self.status = SAC.Status()

        # Create _internals namespace to organize private variables
        self._internals = types.SimpleNamespace()
        self._internals.action_space = env.env.action_space

        # Set temporary variables used to create default networks
        action_space = env.env.action_space
        state_size = np.array(env.env.observation_space.shape).prod()
        action_size = np.array(action_space.shape).prod()
        lows = action_space.low
        highs = action_space.high
        hidden_layers = [int(self.current.hidden_size)] * int(self.current.hidden_depth)

        # And one variable that is public:
        self.status.use_discrete_actions = isinstance(action_space, gym.spaces.Discrete)

        # Create default networks if custom networks weren't provided
        if model is None:
            model = SAC.Model()

        if self.status.use_discrete_actions:
            # Create default networks for discrete actions
            pass
        else:
            # Create default networks for continuous actions
            if model.actor is None:
                mlp = networks.SplitMLP([state_size] + hidden_layers + [action_size], output_splits=2)
                gaussian = networks.NormalActorFromMeanAndStd(mlp)
                model.actor = networks.BoundActor(gaussian, mins=lows, maxs=highs)
            if model.critic1 is None:
                model.critic1 = networks.QMLP(state_size, action_size, hidden_layers)
            if model.critic2 is None:
                model.critic2 = networks.QMLP(state_size, action_size, hidden_layers)

        # Create target networks
        model.critic1_target = deepcopy(model.critic1)
        model.critic2_target = deepcopy(model.critic2)

        # Move networks to the appropriate device
        model.actor = model.actor.to(self.device)
        model.critic1 = model.critic1.to(self.device)
        model.critic2 = model.critic2.to(self.device)
        model.critic1_target = model.critic1_target.to(self.device)
        model.critic2_target = model.critic2_target.to(self.device)

        # Create optimizers
        critic_params = list(model.critic1.parameters()) + list(model.critic2.parameters())
        model.critic_optimizer = optim.Adam(critic_params, lr=self.current.critic_lr)
        model.actor_optimizer = optim.Adam(list(model.actor.parameters()), lr=self.current.actor_lr)

        # Create optimizer required to automatically tune temperature
        if self.current.target_entropy is not None:
            self._internals.log_temperature = nn.Parameter(torch.zeros(1).to(self.device))
            model.entropy_optimizer = optim.Adam([self._internals.log_temperature], lr=self.current.temperature_lr)
            self.hp.temperature = 0

        # Save model in self
        self.model = model

        if checkpoint is not None:
            self._load_checkpoint_second_pass(checkpoint)

    def train(self):
        super().train()

    def _update_learning_rates(self):
        self._update_lr(self.model.actor_optimizer, self.current.actor_lr)
        self._update_lr(self.model.critic_optimizer, self.current.critic_lr)
        if self.current.target_entropy is not None:
            self._update_lr(self.model.entropy_optimizer, self.current.temperature_lr)

    def _update_targets(self):
        """
        Update target networks from live networks, called automatically by train()
        """
        self._update_target(self.model.critic1, self.model.critic1_target)
        self._update_target(self.model.critic2, self.model.critic2_target)

    def _minibatch_update(self, states, actions, next_states, rewards, dones):
        if self.status.use_discrete_actions:
            self._minibatch_update_discrete(states, actions, next_states, rewards, dones)
        else:
            self._minibatch_update_continuous(states, actions, next_states, rewards, dones)

    def _minibatch_update_continuous(self, states, actions, next_states, rewards, dones):
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
            # We compute an unbiased estimate of the values (V_{\bar \psi}(s_{t+1}))
            # of the next states from the Q networks by using an action 
            # sampled from the current policy:
            sampled_actions, sampled_actions_log_p, _ = \
                self.model.actor(next_states)

            # For stability, we estimate Q values with delayed (target) networks, 
            # and take the minimum of the Q values predicted by two target networks. 
            next_q1 = self.model.critic1_target(next_states, sampled_actions).squeeze(-1)
            next_q2 = self.model.critic2_target(next_states, sampled_actions).squeeze(-1)
            next_q = torch.min(next_q1, next_q2)

            # To compute the soft Q value, which includes entropy, we add an unbiased 
            # estimate of the entropy, again computed by sampling:
            next_soft_q = next_q - self.current.temperature * sampled_actions_log_p.mean(-1)

            # Then we force the estimated soft Q value of terminal states to be zero:
            next_soft_q_zeroed = next_soft_q * (1 - dones)

            # Finally, the target Q value is computed with the Bellman equation:
            target_q = rewards + (self.current.gamma * next_soft_q_zeroed)

        # Compute and minimize the critic loss
        # We train both Q networks to predict the target Q value
        q1 = self.model.critic1(states, actions).squeeze(-1)
        q2 = self.model.critic2(states, actions).squeeze(-1)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.model.critic_optimizer.zero_grad()
        q_loss.backward()
        self.model.critic_optimizer.step()

        # Compute and minimize the actor loss
        # Here we maximize the soft Q value
        # (which is entropy, E[-log pi], plus the Q value)
        # by minimizing -1 times the soft Q value
        actions, actions_log_p, actor_entropy = self.model.actor(states)
        q1 = self.model.critic1(states, actions).squeeze(-1)
        q2 = self.model.critic2(states, actions).squeeze(-1)
        min_q = torch.min(q1, q2)
        actor_loss = ((self.current.temperature * actions_log_p.sum(-1)) - min_q).mean()

        self.model.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.model.actor_optimizer.step()

        self.status.entropy = actor_entropy.mean().item()

        # Automatically adjust temperature, see reference 2
        if self.current.target_entropy is not None:
            # Recompute probabilities with new actor parameters
            with torch.no_grad():
                _, log_p, _ = self.model.actor(states)
            # Optimize a Monte Carlo estimate of the loss function
            # from equation 18 in reference 2:
            # J(\alpha) = E_{a_t \tilde \pi_t} [-\alpha \log \pi_t(a_t | s_t) - \alpha \bar H]
            alpha = self._internals.log_temperature.exp()
            temp_loss = (alpha * (-log_p - self.current.target_entropy)).mean()

            self.model.entropy_optimizer.zero_grad()
            temp_loss.backward()
            self.model.entropy_optimizer.step()


    def _random_action(self, state):
        if self.status.use_discrete_actions:
            raise NotImplementedError
        else:
            scale = self._internals.action_space.high - self._internals.action_space.low
            self.status.entropy = np.log(scale).sum()
            return super()._random_action(state)

    def _predict_action(self, state):
        with torch.no_grad():
            action, _, _ = self.model.actor(state)
        return action


    def _save_checkpoint(self):
        filename = self.current.checkpoint_name.format(self=self)
        print("Saving " + filename)
        folder = os.path.dirname(filename)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
        torch.save({
            'hp': self.hp,
            'current': self.current,
            'status': self.status,
            'model_state_dict': self.model.state_dict(),
        }, filename)

    def _load_checkpoint_first_pass(self, checkpoint):
        # First pass, load the hyperparams to properly initialize the agent
        data = torch.load(checkpoint)
        return data['hp']

    def _load_checkpoint_second_pass(self, checkpoint):
        # Second pass, load the checkpoint state
        data = torch.load(checkpoint)
        self.hp = data['hp']
        self.current = data['current']
        self.status = data['status']
        self.model.load_state_dict(data['model_state_dict'])

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
