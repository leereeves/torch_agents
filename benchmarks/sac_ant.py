from torch_agents.agents.sac import SAC
from torch_agents import environments
from torch_agents import schedule

"""
This benchmark replicates Soft Actor Critic performance 
on Open AI Gym Ant environment as reported in:

    Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy 
    deep reinforcement learning with a stochastic actor." International 
    conference on machine learning. PMLR, 2018.

That paper reported results on Ant-V1, but this benchmark uses Ant-V4 because 
V4 uses current Mujoco libary bindings.

The reported average returns are 3500 after 1 million steps 
and 6000 after 3 million.

The reported hyperparameters are:
    2 hidden layers with 256 units each
    (all) learning rates = 3e-4
    discount (gamma) = 0.99
    replay buffer size = 1e6
    target update rate (tau) = 0.005
    target update interval = 1
    gradient steps = 1
    reward scale = 5
"""

if __name__=="__main__":

    env = environments.GymEnv(
            name="Ant-v4", 
            render_mode=None
            )

    hp = SAC.Hyperparams()
    hp.max_actions=3000000
    hp.warmup_actions = 10000

    hp.actor_lr = 3e-4 # schedule.Linear(hp.max_actions, 3e-4, 0).asfloat()
    hp.critic_lr = 3e-4 # schedule.Linear(hp.max_actions, 3e-4, 0).asfloat()

    hp.reward_scale = 5
    hp.temperature = 1

    print(vars(hp))

    agent = SAC(env, hp)
    agent.train()
