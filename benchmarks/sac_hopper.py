from torch_agents.agents.sac import SAC
from torch_agents import environments
from torch_agents import schedule

from pprint import pprint

"""
This benchmark replicates Soft Actor Critic performance 
on Open AI Gym Hopper environment as reported in:

    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy 
    deep reinforcement learning with a stochastic actor." International 
    conference on machine learning. PMLR, 2018.

That paper reported results on Hopper-V1, but this benchmark uses Hopper-V4 
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
    reward scale = 5

The reported average returns are slightly over 3000 (3 points per step)
after 1 million steps.

This implementation achieves 3 points per step, but has a tendency
to fall over, terminating the episode early. I have not found
hyperparameters that work well for this task.
"""
if __name__=="__main__":

    env = environments.GymEnv(
            name="Hopper-v4", 
            render_mode=None
            )

    hp = SAC.Hyperparams()
    hp.max_actions=1000000
    hp.warmup_actions = 10000

    hp.actor_lr = 3e-4
    hp.critic_lr = 3e-4

    hp.reward_scale = 5
    hp.temperature = 1

    pprint(vars(hp))

    agent = SAC(env, hp)
    agent.train()
