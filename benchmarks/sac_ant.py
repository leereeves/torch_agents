from torch_agents.agents.sac import ContinuousSAC
from torch_agents import environments
from torch_agents import schedule

# Soft Actor Critic tuned benchmark for Open AI Gym environment Ant-V4
if __name__=="__main__":

    env = environments.GymEnv(
            name="Ant-v4", 
            render_mode=None
            )

    hp = ContinuousSAC.Hyperparams()
    hp.max_actions=3000000
    hp.actor_lr = schedule.Linear(hp.max_actions, 3e-4, 0).asfloat()
    hp.critic_lr = schedule.Linear(hp.max_actions, 3e-4, 0).asfloat()
    hp.temperature = schedule.Linear(hp.max_actions, 0.2, 0).asfloat()
    hp.warmup_actions = 10000

    agent = ContinuousSAC(env, hp)
    agent.train()
