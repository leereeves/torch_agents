from torch_agents.agents.sac import SAC
from torch_agents import environments
from torch_agents import schedule

from pprint import pprint

# Soft Actor Critic benchmark for Open AI Gym environment Humanoid-V4
# With hyperparameters from the SAC paper
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
