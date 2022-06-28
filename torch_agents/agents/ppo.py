import datetime
import gym
import math
import numpy as np
import random
import time
import torch
import torch.multiprocessing as mp

from copy import deepcopy
from os.path import exists
from torch.utils.tensorboard import SummaryWriter 

from .agent import *

from .. import memory, networks

################################################################################
# Proximal Policy Optimization (PPO)

## References
# Schulman, John, et al. "Proximal policy optimization algorithms." 
# arXiv preprint arXiv:1707.06347 (2017).

class ppoAgent(torch.nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        raise NotImplementedError

class ppo(Agent):
    def __init__(self, name, envs, actor_net, critic_net,
                    device=None, 
                    opt=None,
                    lr=None,
                    critic_coef=0.5,
                    beta=0.1,
                    gamma=0.99,
                    lambd=0.97,
                    epsilon=0.2, # how close to clip the policy gradient
                    clip_neg=None,
                    clip_pos=None,
                    max_epochs=50,
                    steps_per_epoch=1000,
                    updates_per_epoch=100,
                    checkpoint_filename=None,
                    ):

        super().__init__(device)

        self.name = name
        if isinstance(envs, (list, tuple)):
            self.envs = envs
        else:
            self.envs = [envs]
        self.env_count = len(self.envs)

        self.beta = beta
        self.current_beta = 0
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.clip_neg = clip_neg if clip_neg is not None else epsilon
        self.clip_pos = clip_pos if clip_pos is not None else epsilon
        self.max_grad_norm = 0.5
        self.critic_coef = critic_coef

        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.updates_per_epoch = updates_per_epoch

        # Create actor and critic
        self.critic = networks.SqueezeNet(critic_net).to(self.device)

        # Create online memory
        self.memory = [memory.OnPolicyAdvantageMemory(self.steps_per_epoch, self.gamma, self.lambd) for i in range(self.env_count)]

        action_space = self.envs[0].env.action_space
        if isinstance(action_space, gym.spaces.Box):
            self.actor = networks.NormalDistFromMeanNet(actor_net, action_space.shape[0]).to(self.device)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.actor = networks.CategoricalDistFromLogitsNet(actor_net).to(self.device)

        self.agent = ppoAgent(self.actor, self.critic)

        if opt is not None:
            self.opt = opt
            self.lr = None
        else:
            assert(lr is not None)
            # learning rate will be updated before each backprop
            self.opt = torch.optim.Adam(self.agent.parameters(), lr = 0)
            self.lr = lr

        self.action_count = 0
        self.update_count = 0

        self.checkpoint_filename = checkpoint_filename

        # Load old weights if they exist, to continue training
        filename = self.get_model_filename()
        if(filename is not None and exists(filename)):
            t = torch.load(filename, map_location='cpu')
            if t:
                print("Resuming training from existing model")
                #self.policy_network.load_state_dict(t)
        
    def get_model_filename(self):
        return self.checkpoint_filename

    def to_numpy(self, tensors):
        if not isinstance(tensors, (list, tuple)):
            return tensors.cpu().numpy()
        else:
            result = []
            for x in tensors:
                result.append(x.cpu().numpy())
            return tuple(result)

    def to_tensor(self, x, dtype = np.float32):
        return torch.tensor(np.asarray(x, dtype=dtype)).to(self.device)

    def to_tensors(self, *inputs, dtype = np.float32):
        result = []
        for x in inputs:
            result.append(self.to_tensor(x))
        return tuple(result)

    def update(self):
        self.update_count += 1

        # Update learning rates, which can be dynamic parameters
        if self.lr is not None:
            for g in self.opt.param_groups:
                g['lr'] = float(self.lr)

        # Get all data for this epoch from memory 
        all_states = torch.FloatTensor().to(self.device)
        all_actions = torch.FloatTensor().to(self.device)
        all_returns = torch.FloatTensor().to(self.device)
        all_advantages = torch.FloatTensor().to(self.device)
        all_old_logps = torch.FloatTensor().to(self.device)

        for i in range(self.env_count):
            env_states, env_actions, _, _, _, env_returns, env_advantages, env_old_logps = \
                self.to_tensors(*self.memory[i].get())
            all_states = torch.cat((all_states, env_states))
            all_actions = torch.cat((all_actions, env_actions))
            all_returns = torch.cat((all_returns, env_returns))
            all_advantages = torch.cat((all_advantages, env_advantages))
            all_old_logps = torch.cat((all_old_logps, env_old_logps))

        # Save value of beta for this epoch
        self.current_beta = float(self.beta)
        def compute_actor_loss():
            _, logps, entropy = self.agent.actor(states, actions)
            self.entropies.append(entropy.mean().item())
            ratio = torch.exp(logps - old_logps)
            # For transitions with positive advantage,
            # training will increase the ratio, but
            # no further than 1+self.clip_pos
            # For transitions with negative advantage,
            # training will decrease the ratio, but
            # no further than 1-self.clip_neg
            clipped = torch.clamp(ratio, 1-self.clip_neg, 1+self.clip_pos) * advantages
            adv_loss = -(torch.minimum(ratio * advantages, clipped)).mean()
            ent_loss = -(self.current_beta * entropy).mean()
            return adv_loss + ent_loss

        def compute_critic_loss():
            return 0.5 * ((self.agent.critic(states) - returns)**2).mean()

        self.actor_loss = []
        self.critic_loss = []
        self.entropies = []

        data_size = all_states.shape[0]
        minibatch_size = 32
        for i in range(self.updates_per_epoch):
            all_indexes = np.arange(data_size)
            np.random.shuffle(all_indexes)

            for start in range(0, data_size, minibatch_size):
                # Prepare minibatch
                end = start + minibatch_size
                minibatch_indexes = all_indexes[start:end]
                states = all_states[minibatch_indexes]
                actions = all_actions[minibatch_indexes]
                returns = all_returns[minibatch_indexes]
                advantages = all_advantages[minibatch_indexes]
                old_logps = all_old_logps[minibatch_indexes]

                self.opt.zero_grad()
                actor_loss = compute_actor_loss()
                self.actor_loss.append(actor_loss.item())

                critic_loss = compute_critic_loss()
                self.critic_loss.append(critic_loss.item())

                loss = actor_loss + self.critic_coef * critic_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.opt.step()

    def save_model(self):
        #filename = self.get_model_filename()
        #if filename is not None:
        #    torch.save(self.policy_network.state_dict(), filename)
        pass

    def on_end_of_episode(self, score):
        elapsed_time = math.ceil(time.time() - self.start_time)

        self.tb_log.add_scalars(self.name, {'score': score}, self.episode)

        print("Time {}. Episode {}. Action {}. Score {:0.0f}. Avg loss={:.2e} {:.2e}. LR={:.2e} entropy={:.2g}".format(
            datetime.timedelta(seconds=elapsed_time), 
            self.episode, 
            self.action_count, 
            score, 
            np.average(self.actor_loss),
            np.average(self.critic_loss),
            self.opt.param_groups[0]['lr'],
            np.average(self.entropies)
            ))

        if self.episode > 0 and self.episode % 10 == 0:
            print("Saving model {}".format(self.get_model_filename()))
            self.save_model()

        self.episode += 1
        self.score_historys.append(score)

    def train(self):

        # Open Tensorboard log
        path = "./tensorboard/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.tb_log = SummaryWriter(path)

        self.start_time = time.time()
        self.episode = 0

        self.actor_loss = [0]
        self.critic_loss = [0]
        self.entropies = [0]

        self.score_historys = [[] for i in range(self.env_count)]
        scores = [0] * self.env_count
        states = [None] * self.env_count

        for i in range(len(self.envs)):
            states[i] = self.envs[i].reset()

        for self.epoch in range(int(self.max_epochs)):
            for self.step_in_epoch in range(int(self.steps_per_epoch)):
                # For each step, act in every environment and record transition
                for i in range(len(self.envs)):
                    with torch.no_grad():
                        state_tensor = self.to_tensor(states[i])
                        action, logp, _ = self.to_numpy(self.actor(state_tensor))
                        value = self.to_numpy(self.critic(state_tensor))
                    new_state, reward, done, info, life_lost = self.envs[i].step(action)
                    self.action_count += 1
                    scores[i] += reward
                    self.memory[i].store_transition(states[i], action, new_state, reward, done or life_lost, value, logp)
                    if(done):
                        states[i] = self.envs[i].reset()
                        self.on_end_of_episode(scores[i])
                        scores[i] = 0
                        with torch.no_grad():
                            state_tensor = self.to_tensor(states[i])
                            value = self.to_numpy(self.critic(state_tensor))
                            self.memory[i].end_episode(value)
                    else:
                        states[i] = new_state
                # end for i in range(len(self.envs)):
            # end for self.step_in_epoch in range(int(self.steps_per_epoch)):

            # Update networks and reset all memory
            self.update()

            for i in range(len(self.envs)):
                self.memory[i].reset()

        self.env.close()
        self.tb_log.close()

        return self.score_historys
