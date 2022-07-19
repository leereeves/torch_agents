from torch_agents.agents.sac import SAC
from torch_agents import environments
from torch_agents import schedule

# Soft Actor Critic benchmark for Open AI Gym environment Humanoid-V4
# With hyperparameters from the SAC paper
if __name__=="__main__":

    env = environments.GymEnv(
            name="Humanoid-v4", 
            render_mode=None
            )

    hp = SAC.Hyperparams()
    hp.max_actions=1e7
    hp.warmup_actions = 2e5

    hp.actor_lr = schedule.Linear(hp.max_actions, 3e-4, 0).asfloat()
    hp.critic_lr = schedule.Linear(hp.max_actions, 3e-4, 0).asfloat()

    hp.reward_scale = 20
    hp.temperature = 1 # schedule.Linear(hp.max_actions, 1, 0).asfloat()

    agent = SAC(env, hp)
    agent.train()
