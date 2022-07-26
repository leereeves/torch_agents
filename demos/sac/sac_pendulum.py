# Demo of Soft Actor Critic with OpenAI Gym's Pendulum-v1 environment
#
# This is a fairly simple environment that SAC can learn in 
# a few minutes (if not rendering to the screen and therefore
# not limited by the environment's frame rate).
#
# This also demonstrates that annealing learning rate and temperature 
# (the weight of entropy relative to rewards) down to zero is
# effective and results in a precise solution for this environment. 
# In some more complex environments, especially environments that
# terminate early if the agent fails (for example, the agent falls down)
# annealing like this can cause unrecoverable catastrophic forgetting.

import torch_agents as ta
from torch_agents.agents.sac import SAC
from pprint import pprint

if __name__=="__main__":
    env = ta.environments.GymEnv(
            name="Pendulum-v1", 
            render_mode=None # change to "Human" to watch
            )

    hp = SAC.Hyperparams()
    hp.max_actions=3e4
    hp.warmup_actions = 3e3
    hp.actor_lr = ta.schedule.Linear(hp.max_actions, 3e-4, 0)
    hp.critic_lr = ta.schedule.Linear(hp.max_actions, 3e-4, 0)
    hp.temperature_lr = ta.schedule.Linear(hp.max_actions, 1e-3, 0)
    hp.target_entropy = ta.schedule.Linear(hp.max_actions, 0.2, 0)

    pprint(vars(hp))

    agent = SAC(env, hp)
    agent.train()
