# Demo of the simplest possible SAC script

# Because it does not provide hyperparameters,
# this script uses the defaults, which are 
# based on the hyperparameters reported for 
# most environments by:
#
#    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy 
#    deep reinforcement learning with a stochastic actor." International 
#    conference on machine learning. PMLR, 2018.

import torch_agents as ta
from torch_agents.agents.sac import SAC

if __name__=="__main__":
    env = ta.environments.GymEnv("Pendulum-v1")
    agent = SAC(env)
    agent.train()
