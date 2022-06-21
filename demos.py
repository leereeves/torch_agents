import json
import sys
import torch

from torch_agents import agents
from torch_agents import memory
from torch_agents import networks
from torch_agents import explore
from torch_agents import environments
from torch_agents import schedule

# Demo of Double DQN with priority replay memory playing CartPole
def cartpole_demo():
    replay_start_frames = 1000

    env = environments.GymEnv(name="CartPole-v1", render_mode="human", valid_actions=[0,1])
    net = networks.FCNetwork([4, 32, 32, 2])
    mem = memory.PrioritizedReplayMemory(1e6)
    exp = explore.EpsilonGreedyStrategy(
        epsilon=schedule.Sequence([
            schedule.Flat(replay_start_frames, 1.0),
            schedule.Linear(6000, 1.0, 0.4),
            schedule.Sequence(
                [
                    schedule.Linear(3000, 0.4, 0.1),
                ],
                repeat=400
            )
        ]).asfloat(),
        eval_epsilon=0.01,
        min_repeat=1,
        max_repeat=schedule.Sequence(
            [
                schedule.Flat(1000, 3),
                schedule.Flat(1000, 2),
                schedule.Flat(2000, 1)
            ],
            repeat=300
        ).asfloat()
    )
    lr_schedule = schedule.Linear(1e6, 0.0001, 0.00001)
    target_update_schedule = schedule.Linear(10, 1000, 10000)
    agent = agents.dqn('CartPole', env, net, mem, exp, 
        lr = lr_schedule.asfloat(), 
        target_update_freq=target_update_schedule.asfloat(), 
        replay_start_frames=replay_start_frames,
        max_episodes=1e6)
    agent.train()

# Demo of Double DQN with priority replay memory playing Breakout
def breakout_demo():
    env = environments.GymAtariEnv(
            name="BreakoutDeterministic-v4", 
            render_mode=None, 
            valid_actions=[1,2,3],
            action_repeat=4
            )
    net = networks.Mnih2015Atari(3)
    mem = memory.PrioritizedReplayMemory(1e6)
    exp = explore.EpsilonGreedyStrategy(
            epsilon=schedule.Sequence([
                schedule.Flat(50000 // 4, 1.0),
                schedule.Linear(1e6, 1.0, 0.1),
            ]).asfloat(),
            eval_epsilon=0.01
            )
    agent = agents.dqn('Breakout', env, net, mem, exp, lr = 0.00005,
            max_episodes=10000,
            action_repeat=4,
            update_freq=4,
            replay_start_frames=50000,
            target_update_freq=10000,
            checkpoint_filename="breakout.pt"
    )
    agent.train()

# Demo of DDPG with simple replay memory learning Pendulum
def pendulum_demo():
    env = environments.GymEnv(
            name="Pendulum-v1", 
            render_mode=None # "human"
            )
    state_size = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.shape[0]

    actor_net = networks.ActorNetwork(state_size, num_actions)
    critic_net = networks.CriticNetwork(state_size, num_actions)
    mem = memory.ReplayMemory(1e6)
    noise = explore.OrnsteinUhlenbeckProcess(
            sigma=schedule.Linear(50000, 0.2, 0).asfloat(),
            theta=0.15,
            mu=0.0,
            size=num_actions
            )
    agent = agents.ddpg('Pendulum', env, actor_net, critic_net, mem, noise, 
            actor_lr = 0.001,
            critic_lr = 0.001,
            max_episodes=1000,
            action_repeat=1,
            update_freq=1,
            replay_start_frames=200,
            target_update_tau=0.001
    )
    agent.train()

# Demo of DDPG with simple replay memory learning Bipedal Walker
def bipedalwalker_demo():
    env = environments.GymEnv(
            name="BipedalWalker-v3", 
            render_mode=None # "human"
            )
    state_size = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.shape[0]

    actor_net = networks.ActorNetwork(state_size, num_actions)
    critic_net = networks.CriticNetwork(state_size, num_actions)
    mem = memory.ReplayMemory(1e6)
    noise = explore.GaussianNoise(
            mu=0.0,
            sigma=schedule.Linear(500000, 0.2, 0).asfloat(),
            size=num_actions
            )
    agent = agents.ddpg('BipedalWalker', env, actor_net, critic_net, mem, noise, 
            actor_lr = 0.0001,
            critic_lr = 0.001,
            max_episodes=10000,
            action_repeat=1,
            update_freq=1,
            replay_start_frames=2000,
            target_update_tau=0.001
    )
    agent.train()

# Demo of DDPG with simple replay memory learning Hopper
def hopper_demo():
    env = environments.GymEnv(
            name="Hopper-v4", 
            render_mode=None # "human"
            )
    state_size = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.shape[0]

    actor_net = networks.ActorNetwork(state_size, num_actions)
    critic_net = networks.CriticNetwork(state_size, num_actions)
    mem = memory.ReplayMemory(1e6)
    noise = explore.GaussianNoise(
            mu=0.0,
            sigma=schedule.Linear(500000, 0.2, 0).asfloat(),
            size=num_actions
            )
    agent = agents.ddpg('Hopper', env, actor_net, critic_net, mem, noise, 
            actor_lr = 0.0001,
            critic_lr = 0.001,
            max_episodes=10000,
            action_repeat=1,
            update_freq=1,
            replay_start_frames=2000,
            target_update_tau=0.001
    )
    agent.train()

# Demo of DDPG with simple replay memory learning Humanoid
def humanoid_demo():
    env = environments.GymEnv(
            name="Humanoid-v4", 
            render_mode=None # "human"
            )
    state_size = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.shape[0]

    actor_net = networks.ActorNetwork(state_size, num_actions)
    critic_net = networks.CriticNetwork(state_size, num_actions)
    mem = memory.ReplayMemory(1e6)
    noise = explore.GaussianNoise(
            mu=0.0,
            sigma=schedule.Linear(500000, 0.2, 0).asfloat(),
            size=num_actions
            )
    agent = agents.ddpg('Humanoid', env, actor_net, critic_net, mem, noise, 
            actor_lr = 0.0001,
            critic_lr = 0.001,
            max_episodes=10000,
            action_repeat=1,
            update_freq=1,
            replay_start_frames=2000,
            target_update_tau=0.001
    )
    agent.train()

# Main entry point
if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python demos.py <demo name>")
        exit()

    request = sys.argv[1].casefold()

    if request == 'cartpole':
        cartpole_demo()

    if request == 'breakout':
        breakout_demo()

    if request == 'pendulum':
        pendulum_demo()

    if request == 'bipedalwalker':
        bipedalwalker_demo()

    if request == 'hopper':
        hopper_demo()

    if request == 'humanoid':
        humanoid_demo()
