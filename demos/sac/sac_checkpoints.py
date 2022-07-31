# Demo of Soft Actor Critic with OpenAI Gym's Pendulum-v1 environment
#
# This demonstrates the use of checkpoints

import sys
import torch_agents as ta
from torch_agents.agents.sac import SAC
from pprint import pprint

if __name__=="__main__":

    env = ta.environments.GymEnv(name="Pendulum-v1")

    hp = SAC.Hyperparams()
    hp.max_actions=1e4
    hp.warmup_actions = 1e3
    hp.actor_lr = ta.schedule.Linear(hp.max_actions, 3e-4, 0)
    hp.critic_lr = ta.schedule.Linear(hp.max_actions, 3e-4, 0)
    hp.temperature_lr = ta.schedule.Linear(hp.max_actions, 1e-3, 0)
    hp.target_entropy = ta.schedule.Linear(hp.max_actions, 0.2, 0)

    hp.seed = 1
    hp.checkpoint_freq = 1e3
    hp.checkpoint_name = "checkpoints/pendulum{self.status.action_count}.pt"
    
    pprint(vars(hp))

    # First train the agent from scratch and save checkpoints
    agent = SAC(env, hp)
    #agent.train()

    pprint(vars(agent.status))

    # Now load a checkpoint and resume training
    agent = SAC(env, checkpoint="checkpoints/pendulum5000.pt")
    agent.train()

    pprint(vars(agent.status))
