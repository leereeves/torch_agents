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

from . import memory

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def hard_update(target, source):
    for t,s  in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)

################################################################################
# Deep Q Networks

## References

# Brockman, Greg, et al. "Openai gym." arXiv preprint arXiv:1606.01540 (2016).
# https://arxiv.org/pdf/1606.01540.pdf
# https://github.com/openai/gym
# https://www.gymlibrary.ml/

# He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level 
# performance on imagenet classification." Proceedings of the IEEE 
# international conference on computer vision. 2015.
# https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf

# Mnih, Volodymyr, et al. "Human-level control through deep reinforcement 
# learning." nature 518.7540 (2015): 529-533.
# https://daiwk.github.io/assets/dqn.pdf

# Schaul, Tom, et al. "Prioritized experience replay."  ICLR 2016 (2016).
# https://arxiv.org/pdf/1511.05952.pdf

# Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement 
# learning with double q-learning." Proceedings of the AAAI conference 
# on artificial intelligence. Vol. 30. No. 1. 2016.
# https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154

# Zhang, Shangtong, and Richard S. Sutton. "A deeper look at experience 
# replay." arXiv preprint arXiv:1712.01275 (2017).
# https://arxiv.org/pdf/1712.01275.pdf

class dqn(object):
    def __init__(self, name, env, net, mem, exp, 
                    device=None, 
                    opt=None,
                    lr=None, 
                    batch_size=32,
                    gamma=0.99,
                    action_repeat=1,
                    update_freq=1,
                    replay_start_frames=1000,
                    target_update_freq=2000,
                    max_episodes=1000,
                    checkpoint_filename=None,
                    ):

        self.name = name
        self.env = env
        self.strategy = exp
        self.memory = mem
        
        if device is not None:
            self.device = device
        else:        
            if torch.cuda.is_available():
                print("Training on GPU.")
                self.device = torch.device('cuda:0')
            else:
                print("No CUDA device found, or CUDA is not installed. Training on CPU.")
                self.device = torch.device('cpu')

        self.policy_network = net.to(self.device)
        self.target_network = deepcopy(net).to(self.device)

        self.batch_size = batch_size
        self.gamma = gamma
        self.action_repeat = action_repeat
        self.update_freq = update_freq
        self.replay_start_frames = replay_start_frames
        self.target_update_freq = target_update_freq
        self.next_target_update = int(self.target_update_freq)
        self.max_episodes = max_episodes

        self.eval_episode_count = 10
        self.importance_annealing_steps = 1e5

        self.alpha = 1e-5

        if opt is not None:
            self.optimizer = opt
            self.learning_rate = None
        else:
            assert(lr is not None)
            # learning rate will be updated before each backprop
            self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = 0)
            self.learning_rate = lr

        self.loss = torch.nn.SmoothL1Loss(reduction = 'none', beta = 1.0)

        self.action_count = 0
        self.update_count = 0

        self.checkpoint_filename = checkpoint_filename

        # Load old weights if they exist, to continue training
        filename = self.get_model_filename()
        if(filename is not None and exists(filename)):
            t = torch.load(filename, map_location='cpu')
            if t:
                print("Resuming training from existing model")
                self.policy_network.load_state_dict(t)
        
        self.target_network.load_state_dict(self.policy_network.state_dict())


    def get_model_filename(self):
        return self.checkpoint_filename


    def choose_action(self, state):
        is_eval = (self.episode % self.eval_episode_count) == 0
        action = self.strategy.action_without_prediction(self.env.action_space_size(), is_eval)

        if action is None:
            state_tensor = torch.tensor(np.asarray(state, dtype = np.float32)).to(self.device)
            state_tensor = state_tensor.unsqueeze(0) # Add a batch dimension of length 1
            q = self.policy_network.forward(state_tensor)
            q = q.squeeze(0) # remove batch dimension
            action = self.strategy.action_with_prediction(self.env.action_space_size(), is_eval, q)
            self.qs.append(q[action].item())

        return action

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
        self.next_target_update -= 1
        if self.next_target_update <= 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.next_target_update = int(self.target_update_freq)

        indexes, batch, weights = self.memory.sample(self.batch_size)
        states, actions, new_states, rewards, dones = list(zip(*batch))

        # Follow PyTorch's advice:
        # "UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Please consider converting the list to a single numpy.ndarray with numpy.array() 
        # before converting to a tensor."
        states = np.asarray(states)
        actions = np.asarray(actions)
        new_states = np.asarray(new_states)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        weights = np.asarray(weights)

        # Now create tensors
        states_batch = torch.tensor(states, dtype = torch.float32).to(self.device)
        new_states_batch = torch.tensor(new_states,dtype = torch.float32).to(self.device)
        actions_batch = torch.tensor(actions, dtype = torch.long).to(self.device)
        rewards_batch = torch.tensor(rewards, dtype = torch.float32).to(self.device)
        dones_batch = torch.tensor(dones, dtype = torch.float32).to(self.device)
        weights_batch = torch.tensor(weights, dtype = torch.float32).to(self.device)

        # Calculate the Bellman equation, setting the value of Q* to zero in states after the task is done
        with torch.no_grad():
            policy_q = self.policy_network(new_states_batch)
            target_q = self.target_network(new_states_batch)
            next_actions = policy_q.argmax(axis = 1)
            next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_batch + torch.mul(self.gamma * next_q, (1 - dones_batch))
            #target = rewards_batch + torch.mul(self.gamma * new_q.max(axis = 1).values, (1 - dones_batch))

        # Calculate the network's current predictions
        prediction = self.policy_network.forward(states_batch).gather(1,actions_batch.unsqueeze(1)).squeeze(1)

        # Calculate annealing factor for importance sampling
        # Grows linearly from 0.4 to 1
        self.beta = 0.4 + (0.6 * min([self.action_count / self.importance_annealing_steps, 1]))

        # Normalize weights so they only scale the update downwards
        weights_batch = weights_batch / weights_batch.max()

        # Update learning rate, which can be a dynamic parameter
        if self.learning_rate is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = float(self.learning_rate)

        # Train the network to predict the results of the Bellman equation        
        self.optimizer.zero_grad()
        loss = self.loss(prediction, target)
        loss = loss * weights_batch.pow(self.beta) # importance sampling
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        # Update weights in prioritized replay memory
        with torch.no_grad():
            deltas = (target-prediction).absolute()
        for i in range(self.batch_size):
            self.memory.update_weight(indexes[i], deltas[i].item() + self.alpha)

        return

    def save_model(self):
        filename = self.get_model_filename()
        if filename is not None:
            torch.save(self.policy_network.state_dict(), filename)

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
            self.qs = [0]
            while not done:
                # This loops through steps (4 frames for Atari, 1 frame for Cartpole)
                action = self.choose_action(state)
                new_state, reward, done, info, life_lost = self.env.step(action)
                self.action_count += 1
                score += reward
                clipped_reward = np.sign(reward)
                self.memory.store_transition(state, action, new_state, clipped_reward, done or life_lost)
                self.minibatch_update()
                state = new_state

            scores.append(score)
            moving_average = np.average(scores[-20:])
            elapsed_time = math.ceil(time.time() - start)

            tb_log.add_scalars(self.name, {'score': scores[-1]}, self.episode)

            print("Time {}. Episode {}. Action {}. Score {:0.0f}. MAvg={:0.1f}. {}. Avg p={:0.2f}. Avg q={:0.2f}. LR={:g}".format(
                datetime.timedelta(seconds=elapsed_time), 
                self.episode, 
                self.action_count, 
                score, 
                moving_average, 
                str(self.strategy), 
                self.memory.tree.get_average_weight(), 
                np.average(self.qs),
                self.optimizer.param_groups[0]['lr']
                ))

            if self.episode > 0 and self.episode % 10 == 0:
                print("Saving model {}".format(self.get_model_filename()))
                self.save_model()

        self.env.close()
        tb_log.close()

################################################################################
# Deep Deterministic Policy Gradients (DDPG)

## References

# Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement 
# learning." arXiv preprint arXiv:1509.02971 (2015).
# https://arxiv.org/abs/1509.02971

class ddpg(object):
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

        self.name = name
        self.env = env
        self.noise = noise
        self.memory = mem
        
        if device is not None:
            self.device = device
        else:        
            if torch.cuda.is_available():
                print("Training on GPU.")
                self.device = torch.device('cuda:0')
            else:
                print("No CUDA device found, or CUDA is not installed. Training on CPU.")
                self.device = torch.device('cpu')

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

    def to_tensor(self, x):
        return torch.tensor(np.asarray(x, dtype = np.float32)).to(self.device)

    def choose_action(self, state):
        is_eval = (self.episode > 0) and ((self.episode % self.eval_episode_count) == 0)

        with torch.no_grad():
            state_tensor = self.to_tensor(state)
            if(len(state_tensor.shape) > 1):
                state_tensor = state_tensor.squeeze(1) # remove weird input dimension
            state_tensor = state_tensor.unsqueeze(0) # Add a batch dimension
            a = self.policy_actor_network.forward(state_tensor)
            a = a.squeeze(0).cpu().numpy() # remove batch dimension and untensor
        
        if not is_eval:
            a += self.noise.sample()

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

        # Now update the critic network
        indexes, batch, weights = self.memory.sample(self.batch_size)
        states, actions, new_states, rewards, dones = list(zip(*batch))

        # Follow PyTorch's advice:
        # "UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Please consider converting the list to a single numpy.ndarray with numpy.array() 
        # before converting to a tensor."
        states = np.asarray(states)
        actions = np.asarray(actions)
        new_states = np.asarray(new_states)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        weights = np.asarray(weights)

        # Now create tensors
        states_batch = torch.tensor(states, dtype = torch.float32).to(self.device)
        new_states_batch = torch.tensor(new_states,dtype = torch.float32).to(self.device)
        actions_batch = torch.tensor(actions, dtype = torch.long).to(self.device)
        rewards_batch = torch.tensor(rewards, dtype = torch.float32).to(self.device)
        dones_batch = torch.tensor(dones, dtype = torch.float32).to(self.device)
        weights_batch = torch.tensor(weights, dtype = torch.float32).to(self.device)

        with torch.no_grad():
            next_q = self.target_critic_network(
                new_states_batch,
                self.target_actor_network(new_states_batch),
            ).squeeze(1)
            goal_q = rewards_batch + torch.mul(self.gamma * next_q, (1 - dones_batch))

        # Critic update
        self.policy_critic_network.zero_grad()
        q = self.policy_critic_network(states_batch, actions_batch).squeeze(1)
        q_loss = self.critic_loss_function(q, goal_q)
        q_loss.backward()
        self.critic_opt.step()
        self.q_loss.append(q_loss.item())

        # Actor update
        self.policy_actor_network.zero_grad()
        policy_loss = -self.policy_critic_network(states_batch, self.policy_actor_network(states_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_opt.step()

        return

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
