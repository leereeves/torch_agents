import math
import numpy as np
import random
import time
import torch
from copy import deepcopy

import torch_agents as ta

class Agent(object):
    class Hyperparams(object):
        """
        Hyperparameters used by all agents. Unless otherwise noted,
        these may be ints, floats, or schedules derived from class Schedule, 
        allowing any hyperparameter to change dynamically during training.
        """
        def __init__(self):
            self.seed = None
            """If None (which is the default) torch-agents will not 
            initialize any random seeds. If this is not None, 
            torch-agents initializes the torch, np, and random seeds 
            and uses the deterministic cuDNN backend. 
            
            .. note::
                The PyTorch authors warn: "Completely reproducible 
                results are not guaranteed across PyTorch releases, 
                individual commits, or different platforms. Furthermore, 
                results may not be reproducible between CPU and GPU 
                executions, even when using identical seeds." 
                
                "Also, deterministic operations are often slower than 
                nondeterministic operations, so single-run performance 
                may decrease for your model. However, determinism may 
                save time in development by facilitating experimentation, 
                debugging, and regression testing."

                https://pytorch.org/docs/stable/notes/randomness.html
            """
            self.checkpoint_name = None
            """
            The name of checkpoint files. This can be a formatted string
            that includes hyperparameters (self.current) and status 
            variables (self.status).
            """
            self.checkpoint_freq = None
            """
            Minimum number of actions between checkpoints. The actual 
            number may be higher because checkpoints are always
            saved at the end of an episode. If checkpoint_name is not 
            None and checkpoint_freq is None, only one checkpoint
            will be saved, when training is finished.
            """
    class Status(object):
        def __init__(self):
            super().__init__()
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
            self.use_discrete_actions = None
            "True if the environment has discrete actions, False if continuous."
            self.last_checkpoint_ac = 0
            "Action count when the most recent checkpoint was saved."

    def __init__(self, device, env, hp:Hyperparams):
        self.env = env
        if device is not None:
            self.device = device
        else: # if device is None, autodetect CUDA
            if torch.cuda.is_available():
                print("Training on GPU.")
                self.device = torch.device('cuda:0')
            else:
                print("No CUDA device found, or CUDA is not installed. Training on CPU.")
                self.device = torch.device('cpu')

        self.hp = deepcopy(hp)
        self.current = deepcopy(hp)
        self._update_hyperparams() # to set initial values of scheduled hyperparams

        if hp.seed is not None:
            self._set_seed(hp.seed)

    def train(self):
        "Train the agent."
        pass

    def _set_seed(self, seed):
        print(f"Setting seed {seed:g}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)

    def _save_checkpoint(self):
        raise NotImplementedError


class OffPolicyAgent(Agent):
    class Hyperparams(Agent.Hyperparams):
        """
        Hyperparameters to configure an OffPolicyAgent. Unless otherwise noted,
        these may be ints, floats, or schedules derived from class Schedule, 
        allowing any hyperparameter to change dynamically during training.
        """
        def __init__(self):
            super().__init__()
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


    class Status(Agent.Status):
        def __init__(self):
            super().__init__()
            # at this time, there are no attributes unique to OffPolicyAgents


    ########################################
    # The OffPolicyAgent class starts here 

    def __init__(self, device, env:ta.environments.EnvInterface, hp:Hyperparams):
        super().__init__(device, env, hp)

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
            elif isinstance(value, str):
                setattr(self.current, param_name, value)
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

        state = self.env.reset()
        score = 0
        done = 0
        # This loops through actions (4 frames for Atari, 1 frame for most envs)
        while self.status.action_count < self.current.max_actions:
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

                if self.current.checkpoint_name is not None and \
                    self.current.checkpoint_freq is not None and \
                    self.status.action_count >= \
                    self.status.last_checkpoint_ac + self.current.checkpoint_freq:

                    # save checkpoint
                    self.status.last_checkpoint_ac = self.status.action_count
                    self._save_checkpoint()

                # Reset for the start of a new episode
                state = self.env.reset()
                score = 0
                done = 0
            # end if done

            #if self.episode > 0 and self.episode % 10 == 0:
            #    print("Saving model {}".format(self.get_model_filename()))
            #    self.save_model()

            self._update_hyperparams()

        # end while self.status.action_count < self.current.max_actions:

        # Save final checkpoint. This is saved even if checkpoint_freq
        # is None, if checkpoint_name is not None.
        if self.current.checkpoint_name is not None:
            self.status.last_checkpoint_ac = self.status.action_count
            self._save_checkpoint()

        self.env.close()
        #tb_log.close()
