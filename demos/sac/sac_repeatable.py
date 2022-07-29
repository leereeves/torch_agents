# Demo of Soft Actor Critic with OpenAI Gym's Pendulum-v1 environment
#
# This demonstrates repeatable results when run more than once
# on the same system using the `seed` hyperparameter.
#
# However, the results probably still may not be identical when
# run on different systems.

import sys
import torch_agents as ta
from torch_agents.agents.sac import SAC
from pprint import pprint

if __name__=="__main__":

    # Run "python sac_pendulum.py human" to watch the game
    if len(sys.argv) > 1:
        render_mode = sys.argv[1].casefold()
    else:
        render_mode = None

    env = ta.environments.GymEnv(
            name="Pendulum-v1", 
            render_mode=render_mode
            )

    hp = SAC.Hyperparams()
    hp.max_actions=3e4
    hp.warmup_actions = 3e3
    hp.actor_lr = ta.schedule.Linear(hp.max_actions, 3e-4, 0)
    hp.critic_lr = ta.schedule.Linear(hp.max_actions, 3e-4, 0)
    hp.temperature_lr = ta.schedule.Linear(hp.max_actions, 1e-3, 0)
    hp.target_entropy = ta.schedule.Linear(hp.max_actions, 0.2, 0)

    hp.seed = 1
    
    pprint(vars(hp))

    agent = SAC(env, hp)
    agent.train()
