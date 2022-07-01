import math
import numpy as np
import time
import torch

from ..environments import EnvInterface

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def hard_update(target, source):
    for t,s  in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)

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


class OffPolicyAgent(Agent):
    def __init__(self, device, env:EnvInterface):
        super().__init__(device, env)

    # Convert a Torch tensor to a numpy array,
    # And if that tensor is on the GPU, move to CPU
    def to_numpy(self, tensors):
        return tensors.cpu().numpy()

    # Convert a single object understood by np.asarray to a tensor
    # and move the tensor to our device
    def to_tensor(self, x, dtype = np.float32):
        return torch.tensor(np.asarray(x, dtype=dtype)).to(self.device)

    def update_hyperparams(self):
        for param, value in vars(self.hp).items():
            setattr(self.current, param, float(value))

    def update_lr(self, optimizer, lr):
        if lr is not None:
            for g in optimizer.param_groups:
                g['lr'] = lr

    def update_model(self):

        # Many agents don't start training until the replay
        # buffer has enough samples to reduce correlation 
        # in each minibatch of the training data.
        if self.status.action_count < self.current.warmup_actions:
            return

        # Ensure we have enough memory to sample a full minibatch
        if len(self.memory) < self.current.minibatch_size:
            return

        # Update learning rates, which can be dynamic parameters
        self.update_lr(self.modules.actor_optimizer, self.current.actor_lr)
        self.update_lr(self.modules.critic_optimizer, self.current.critic_lr)

        # Update target networks, possibly on a different 
        # schedule than minibatch updates
        if self.status.action_count % self.current.target_update_freq == 0:
            self.update_targets()

        #####
        # Beyond this point, we begin the parameter update. The code
        # here handles prep steps shared by all agents before
        # calling the agent specific function minibatch_update()

        # Some agents don't update every action
        if self.status.action_count % self.current.update_freq != 0:
            return

        # All requirements are satisfied, it's time for an update
        self.status.update_count += 1

        # Get a sample from the replay memory
        _, batch, _ = self.memory.sample(int(self.current.minibatch_size))
        # Rearrange from list of transitions to lists of states, actions, etc
        list_of_data_lists = list(zip(*batch))
        # Convert to tensors
        states, actions, next_states, rewards, dones = \
            map(self.to_tensor, list_of_data_lists)

        # Update the gradient (every algorithm implements this differently)
        self.minibatch_update(states, actions, next_states, rewards, dones)

    def update_target(self, live, target):
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
            action = self.choose_action(state)
            new_state, reward, done, info, life_lost = self.env.step(action)
            self.status.action_count += 1
            score += reward
            if self.current.clip_rewards:
                clipped_reward = np.sign(reward)
            else:
                clipped_reward = reward
            self.memory.store_transition(state, action, new_state, clipped_reward, done or life_lost)
            self.update_model()
            state = new_state

            self.status.elapsed_time = math.ceil(time.time() - start)

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

            self.update_hyperparams()
        # end for self.status.action_count in range(int(self.current.max_actions)):

        self.env.close()
        #tb_log.close()
