import math
import numpy as np
import time
import torch

import torch_agents as ta

class Agent(object):
    def __init__(self, device, env):
        self.env = env
        if device is not None:
            self.device = device
        else:        
            if torch.cuda.is_available():
                print("Training on GPU.")
                self.device = torch.device('cuda:0')
            else:
                print("No CUDA device found, or CUDA is not installed. Training on CPU.")
                self.device = torch.device('cpu')

    def train(self):
        "Train the agent."
        pass


class OffPolicyAgent(Agent):
    class Hyperparams(object):
        """
        Hyperparameters to configure an SAC agent. These may be ints, floats, or 
        schedules derived from class Schedule, allowing any hyperparameter
        to change dynamically during training.
        """
        def __init__(self):
            self.max_actions = 1e6
            """How many actions to take during training, unless stopped early"""
            self.minibatch_size = 256
            "Number of transitions in each minibatch. Default value is 256."
            self.warmup_actions = 10000
            "Random actions to take before beginning to train the networks."
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

    ########################################
    # The OffPolicyAgent class starts here 

    def __init__(self, device, env:ta.environments.EnvInterface):
        super().__init__(device, env)

    # Convert a Torch tensor to a numpy array,
    # And if that tensor is on the GPU, move to CPU
    def _to_numpy(self, x, dtype = np.float32):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        else:
            return np.asarray(x, dtype=dtype)

    # Convert a single object understood by np.asarray to a tensor
    # and move the tensor to our device
    def _to_tensor(self, x, dtype = np.float32):
        if torch.is_tensor(x):
            return x.to(self.device)
        else:
            return torch.tensor(np.asarray(x, dtype=dtype)).to(self.device)

    def _update_hyperparams(self):
        for param_name, value in vars(self.hp).items():
            if value is None:
                setattr(self.current, param_name, None)
            else:
                setattr(self.current, param_name, float(value))
                if isinstance(value, ta.schedule.Schedule):
                    value.advance()

    def _update_lr(self, optimizer, lr):
        if lr is not None:
            for g in optimizer.param_groups:
                g['lr'] = lr

    def _update_model(self):

        # Many agents don't start training until the replay
        # buffer has enough samples to reduce correlation 
        # in each minibatch of the training data.
        if self.status.action_count < self.current.warmup_actions:
            return

        # Ensure we have enough memory to sample a full minibatch
        if len(self.memory) < self.current.minibatch_size:
            return

        # Update learning rates, which can be dynamic parameters
        self._update_learning_rates()

        # Update target networks, possibly on a different 
        # schedule than minibatch updates
        if self.status.action_count % self.current.target_update_freq == 0:
            self._update_targets()

        #####
        # Beyond this point, we begin the parameter update. The code
        # here handles prep steps shared by all agents before
        # calling the agent specific function minibatch_update()

        # Some agents don't update every action
        if self.status.action_count % self.current.update_freq != 0:
            return

        # All requirements are satisfied
        self.status.update_count += 1
        for _ in range(int(self.current.minibatches_per_update)):
            self.status.minibatch_count += 1
            # Prepare a minibatch:
            #   Get a sample from the replay memory
            _, batch, _ = self.memory.sample(int(self.current.minibatch_size))
            #   Rearrange from list of transitions to lists of states, actions, etc
            list_of_data_lists = list(zip(*batch))
            #   Convert to tensors
            states, actions, next_states, rewards, dones = \
                map(self._to_tensor, list_of_data_lists)

            # Call a derived class function to descend the gradient with this minibatch
            self._minibatch_update(states, actions, next_states, rewards, dones)

    def _update_target(self, live, target):
        tau = self.current.target_update_rate
        for param, target_param in zip(live.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
        for self.status.action_count in range(int(self.current.max_actions)):
            state_tensor = self._to_tensor(state)
            action = self._to_numpy(self._choose_action(state_tensor))
            new_state, reward, done, info, life_lost = self.env.step(action)
            self.status.action_count += 1
            score += reward
            if self.current.clip_rewards:
                adj_reward = np.clip(reward, -1, 1)
            else:
                adj_reward = reward
            adj_reward *= self.current.reward_scale
            self.memory.store_transition(state, action, new_state, adj_reward, done or life_lost)
            state = new_state

            # Update models
            self._update_model()

            # Update status
            self.status.elapsed_time = math.ceil(time.time() - start)

            # Handle end of episode
            if done:
                self.status.episode_count += 1
                self.status.score_history.append(score)

                self.on_episode_end()

                # Reset for the start of a new episode
                state = self.env.reset()
                score = 0
                done = 0
            # end if done

            #if self.episode > 0 and self.episode % 10 == 0:
            #    print("Saving model {}".format(self.get_model_filename()))
            #    self.save_model()

            self._update_hyperparams()
        # end for self.status.action_count in range(int(self.current.max_actions)):

        self.env.close()
        #tb_log.close()
