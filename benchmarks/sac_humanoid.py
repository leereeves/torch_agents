from torch_agents.agents.sac import SAC
from torch_agents import environments
from torch_agents import schedule

from pprint import pprint

"""
This benchmark replicates Soft Actor Critic performance 
on Open AI Gym Humanoid environment as reported in:

    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy 
    deep reinforcement learning with a stochastic actor." International 
    conference on machine learning. PMLR, 2018.

That paper reported results on Humanoid-V1, but this benchmark uses Humanoid-V4 
because V4 uses current Mujoco libary bindings. This may cause some
differences in performance.

The reported hyperparameters are:
    2 hidden layers with 256 units each
    (all) learning rates = 3e-4
    discount (gamma) = 0.99
    replay buffer size = 1e6
    target update rate (tau) = 0.005
    target update interval = 1
    gradient steps = 1
    reward scale = 20

The reported average returns are slightly over 4000 (4 points per step)
after 1 million steps, and slightly over 6000 after 10 million steps.
"""
if __name__=="__main__":

    env = environments.GymEnv(
            name="Humanoid-v4", 
            render_mode=None
            )

    hp = SAC.Hyperparams()
    hp.max_actions=1e7
    hp.warmup_actions = 1e4

    hp.actor_lr = 3e-4
    hp.critic_lr = 3e-4

    hp.reward_scale = 20
    hp.temperature = 1

    pprint(vars(hp))

    agent = SAC(env, hp)
    agent.train()
