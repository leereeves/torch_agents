import datetime
import math
import numpy as np
import random
import time
import torch
import torch.multiprocessing as mp

from copy import deepcopy
from os.path import exists
from torch.utils.tensorboard import SummaryWriter 

from .. import memory

from .agent import *

################################################################################
# Deep Deterministic Policy Gradients (DDPG)

## References

# Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement 
# learning." arXiv preprint arXiv:1509.02971 (2015).
# https://arxiv.org/abs/1509.02971

class ddpg(Agent):
    def __init__(self, name, env, actor_net, critic_net, mem, noise, 
                    device=None, 
                    actor_opt=None,
                    critic_opt=None,
                    actor_lr=None, 
                    critic_lr=None,
                    batch_size=32,
                    gamma=0.99,
                    action_repeat=1,
                    update_freq=1,
                    replay_start_frames=1000,
                    target_update_freq=None,
                    target_update_tau=None,
                    max_episodes=1000,
                    checkpoint_filename=None,
                    ):

        super().__init__(device)

        self.name = name
        self.env = env
        self.noise = noise
        self.memory = mem
        
        self.policy_actor_network = actor_net.to(self.device)
        self.target_actor_network = deepcopy(actor_net).to(self.device)

        self.policy_critic_network = critic_net.to(self.device)
        self.target_critic_network = deepcopy(critic_net).to(self.device)

        self.batch_size = batch_size
        self.gamma = gamma
        self.action_repeat = action_repeat
        self.update_freq = update_freq
        self.replay_start_frames = replay_start_frames
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        if target_update_freq is not None:
            self.next_target_update = int(self.target_update_freq)
        self.max_episodes = max_episodes

        self.eval_episode_count = 10
        self.importance_annealing_steps = 1e5

        self.alpha = 1e-5

        if actor_opt is not None:
            self.actor_opt = actor_opt
            self.actor_lr = None
        else:
            assert(actor_lr is not None)
            # learning rate will be updated before each backprop
            self.actor_opt = torch.optim.Adam(self.policy_actor_network.parameters(), lr = 0)
            self.actor_lr = actor_lr

        if critic_opt is not None:
            self.critic_opt = critic_opt
            self.critic_lr = None
        else:
            assert(critic_lr is not None)
            # learning rate will be updated before each backprop
            self.critic_opt = torch.optim.Adam(self.policy_critic_network.parameters(), lr = 0)
            self.critic_lr = critic_lr


        self.critic_loss_function = torch.nn.MSELoss()

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
        
        hard_update(self.target_actor_network, self.policy_actor_network)
        hard_update(self.target_critic_network, self.policy_critic_network)


    def get_model_filename(self):
        return self.checkpoint_filename

    def to_tensor(self, x, dtype = np.float32):
        return torch.tensor(np.asarray(x, dtype=dtype)).to(self.device)

    def to_tensors(self, *inputs, dtype = np.float32):
        result = []
        for x in inputs:
            result.append(self.to_tensor(x))
        return tuple(result)

    def choose_action(self, state):
        is_eval = (self.episode > 0) and ((self.episode % self.eval_episode_count) == 0)

        # ask the actor network for the best action in this state
        with torch.no_grad():
            state_tensor = self.to_tensor(state).flatten()
            a = self.policy_actor_network.forward(state_tensor)
            a = a.cpu().numpy()

        # always sample the noise so dynamic parameters update on schedule
        noise = self.noise.sample()

        # but only use the noise during training, not eval
        if not is_eval:
            a += noise

        return a

    def minibatch_update(self):
        # "a uniform random policy is run for replay_start_frames frames before learning starts"
        # That is replay_start_frames / action_repeat actions (steps)
        if self.action_count < (self.replay_start_frames / self.action_repeat):
            return

        # "update_freq actions are selected by the agent between successive SGD updates"
        if self.action_count % self.update_freq != 0:
            return

        # Ensure we have enough memory to sample a full batch
        if len(self.memory) < self.batch_size:
            return

        # All requirements are satisfied, it's time for an update
        self.update_count += 1

        # "target_update_freq is the frequency (measured in number of parameter updates) 
        # with which the target network is updated.
        if self.target_update_freq is not None:
            self.next_target_update -= 1
            if self.next_target_update <= 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
                self.next_target_update = int(self.target_update_freq)
        elif self.target_update_tau is not None:
            soft_update(self.target_actor_network, self.policy_actor_network, self.target_update_tau)
            soft_update(self.target_critic_network, self.policy_critic_network, self.target_update_tau)

        # Update learning rates, which can be dynamic parameters
        if self.actor_lr is not None:
            for g in self.actor_opt.param_groups:
                g['lr'] = float(self.actor_lr)
        if self.critic_lr is not None:
            for g in self.critic_opt.param_groups:
                g['lr'] = float(self.critic_lr)

        # Update the critic network
        _, batch, _ = self.memory.sample(self.batch_size)
        states, actions, new_states, rewards, dones = list(zip(*batch))

        # Create tensors
        states, actions, new_states, rewards, dones = \
            self.to_tensors(states, actions, new_states, rewards, dones)

        # Compute the Q target
        with torch.no_grad():
            next_q = self.target_critic_network(
                new_states,
                self.target_actor_network(new_states),
            ).squeeze(1)
            target_q = rewards + torch.mul(self.gamma * next_q, (1 - dones))

        # Train the critic network to predict the Q target
        self.policy_critic_network.zero_grad()
        q = self.policy_critic_network(states, actions).squeeze(1)
        q_loss = self.critic_loss_function(q, target_q)
        q_loss.backward()
        self.critic_opt.step()
        self.q_loss.append(q_loss.item())

        # Train the actor network to predict the action with the best Q value
        self.policy_actor_network.zero_grad()
        policy_loss = -self.policy_critic_network(states, self.policy_actor_network(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_opt.step()

    def save_model(self):
        #filename = self.get_model_filename()
        #if filename is not None:
        #    torch.save(self.policy_network.state_dict(), filename)
        pass

    def train(self):

        # Open Tensorboard log
        path = "./tensorboard/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tb_log = SummaryWriter(path)

        # Log hyperparameters
        #for key, value in self.config.items():
        #    tb_log.add_text(key, str(value))

        start = time.time()
        scores = []

        for self.episode in range(int(self.max_episodes)):
            # This is the start of an episode
            state = self.env.reset()
            score = 0
            done = 0
            self.q_loss = [0]
            while not done:
                # This loops through steps (4 frames for Atari, 1 frame for Cartpole)
                action = self.choose_action(state)
                new_state, reward, done, info, life_lost = self.env.step(action)
                self.action_count += 1
                score += reward
                #clipped_reward = np.sign(reward)
                self.memory.store_transition(state, action, new_state, reward, done or life_lost)
                self.minibatch_update()
                state = new_state

            scores.append(score)
            moving_average = np.average(scores[-20:])
            elapsed_time = math.ceil(time.time() - start)

            tb_log.add_scalars(self.name, {'score': scores[-1]}, self.episode)

            print("Time {}. Episode {}. Action {}. Score {:0.0f}. MAvg={:0.1f}. {}. Avg q_loss={:g}. LR={:g}".format(
                datetime.timedelta(seconds=elapsed_time), 
                self.episode, 
                self.action_count, 
                score, 
                moving_average, 
                str(self.noise), 
                np.average(self.q_loss),
                self.actor_opt.param_groups[0]['lr']
                ))

            if self.episode > 0 and self.episode % 10 == 0:
                print("Saving model {}".format(self.get_model_filename()))
                self.save_model()

        self.env.close()
        tb_log.close()
