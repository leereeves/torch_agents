import json
import numpy as np
import sys
import torch

from torch_agents.agents.dqn import dqn
from torch_agents.agents.ddpg import ddpg
from torch_agents.agents.ppo import ppo
from torch_agents.agents.sac import SAC
from torch_agents import memory
from torch_agents import networks
from torch_agents import explore
from torch_agents import environments
from torch_agents import schedule

# Demo of Double DQN with priority replay memory playing CartPole
def dqn_cartpole_demo():
    replay_start_frames = 1000

    env = environments.GymEnv(name="CartPole-v1", render_mode="human", valid_actions=[0,1])
    net = networks.MLP([4, 32, 32, 2])
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
        ]),
        eval_epsilon=0.01,
        min_repeat=1,
        max_repeat=schedule.Sequence(
            [
                schedule.Flat(1000, 3),
                schedule.Flat(1000, 2),
                schedule.Flat(2000, 1)
            ],
            repeat=300
        )
    )
    lr_schedule = schedule.Linear(1e6, 0.0001, 0.00001)
    target_update_schedule = schedule.Linear(10, 1000, 10000)
    agent = dqn('CartPole', env, net, mem, exp, 
        lr = lr_schedule, 
        target_update_freq=target_update_schedule, 
        replay_start_frames=replay_start_frames,
        max_episodes=1e6)
    agent.train()

# Demo of Double DQN with priority replay memory playing Breakout
def dqn_atari_demo(env_name):
    env = environments.GymAtariEnv(
            name=env_name, 
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
            ]),
            eval_epsilon=0.01
            )
    agent = dqn(env_name, env, net, mem, exp, lr = 0.00005,
            max_episodes=10000,
            action_repeat=4,
            update_freq=4,
            replay_start_frames=50000,
            target_update_freq=10000,
            checkpoint_filename=None
    )
    agent.train()

# Demo of DDPG with simple replay memory
def ddpg_demo(env_name, render_mode=None):
    env = environments.GymEnv(
            name=env_name, 
            render_mode=render_mode
            )
    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.shape[0]

    actor_net = networks.ContinuousActorNetwork(state_size, action_size)
    critic_net = networks.QMLP(state_size, action_size, [400,300])
    mem = memory.ReplayMemory(1e6)
    noise = explore.GaussianNoise(
            mu=0.0,
            sigma=schedule.Linear(500000, 0.2, 0),
            size=action_size
            )
    agent = ddpg(env_name, env, actor_net, critic_net, mem, noise, 
            actor_lr = 0.0001,
            critic_lr = 0.001,
            max_episodes=10000,
            action_repeat=1,
            update_freq=1,
            replay_start_frames=2000,
            target_update_tau=0.001
    )
    agent.train()

# Demo of Proximal Policy Optimization
def ppo_demo(env_name, render_mode=None):
    env = environments.GymEnv(
            name=env_name, 
            render_mode=render_mode
            )
    state_size = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.shape[0]

    max_steps=1000000
    steps_per_epoch=2000
    max_epochs=max_steps//steps_per_epoch

    actor_net = networks.ContinuousActorNetwork(state_size, num_actions, activation=torch.nn.Tanh)
    critic_net = networks.MLP([state_size, 64, 64, 1], activation=torch.nn.Tanh)
    agent = ppo(env_name, env, actor_net, critic_net, 
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            updates_per_epoch=10,
            lr = schedule.Linear(max_epochs, 3e-4, 0),
            beta = 0,
            gamma = 0.99,
            lambd = 0.95,
    )
    agent.train()

def grid_search(f, **grid):
    keys = []
    values = []
    for k, v in grid.items():
        keys.append(k)
        values.append(v)
    lengths = [len(v) for v in values]
    divisors = [np.prod(lengths[0:x]) for x in range(0, len(lengths))]
    grid_size = np.prod(lengths)

    kwarg_history = []
    score_history = []
    for i in range(grid_size):
        kwargs = {}        
        for j, key in enumerate(keys):
            grid_coordinate = int((i//divisors[j]) % lengths[j])
            kwargs[keys[j]] = values[j][grid_coordinate]
        print("============================")
        print(kwargs)
        print("============================")
        scores = f(**kwargs)
        kwarg_history.append(kwargs)
        score_history.append(np.average(scores[-100:]))

    for i in range(len(kwarg_history)):
        print(kwarg_history[i])
        print(score_history[i])



# Demo of Proximal Policy Optimization
def ppo_cartpole_demo(render_mode=None):
    env_name = "CartPole-v1"

    envs = [environments.GymEnv(
            name=env_name, 
            render_mode=render_mode,
            valid_actions=[0,1]
            )
            for _ in range(4)]
    state_size = envs[0].env.observation_space.shape[0]

    max_steps=500000
    steps_per_epoch=128
    max_epochs=max_steps//steps_per_epoch

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    actor_net = torch.nn.Sequential(
        layer_init(torch.nn.Linear(state_size, 64)),
        torch.nn.Tanh(),
        layer_init(torch.nn.Linear(64, 64)),
        torch.nn.Tanh(),
        layer_init(torch.nn.Linear(64, 2), std=0.01),
    )
    critic_net = torch.nn.Sequential(
        layer_init(torch.nn.Linear(state_size, 64)),
        torch.nn.Tanh(),
        layer_init(torch.nn.Linear(64, 64)),
        torch.nn.Tanh(),
        layer_init(torch.nn.Linear(64, 1), std=1.0),
    )
    #actor_net = networks.MLP([state_size, 64, 64, 2], activation=torch.nn.Tanh)
    #critic_net = networks.MLP([state_size, 64, 64, 1], activation=torch.nn.Tanh)
    agent = ppo(env_name, envs, actor_net, critic_net, 
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            updates_per_epoch=4,
            lr = schedule.Linear(max_epochs, 2.5e-4, 0),
            gamma = 0.99,
            lambd = 0.95,
            beta = 0.01,
            epsilon = 0.2,
            critic_coef = 0.5
    )
    return agent.train()

# Main entry point
if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python demos.py <demo name>")
        exit()

    request = sys.argv[1].casefold()

    if request == 'dqn-cartpole':
        dqn_cartpole_demo()

    if request == 'dqn-breakout':
        dqn_atari_demo("BreakoutDeterministic-v4")

    if request == 'dqn-pong':
        dqn_atari_demo("ALE/Pong-v5")

    if request == 'ddpg-pendulum':
        ddpg_demo("Pendulum-v1")

    if request == 'ddpg-bipedalwalker':
        ddpg_demo("BipedalWalker-v3")

    if request == 'ddpg-hopper':
        ddpg_demo("Hopper-v4")

    if request == 'ddpg-humanoid':
        ddpg_demo("Humanoid-v4")

    if request == 'ddpg-pendulum':
        ddpg_demo("Pendulum-v1")

    if request == 'ppo-pendulum':
        ppo_demo("Pendulum-v1")

    if request == 'ppo-humanoid':
        ppo_demo("Humanoid-v4")

    if request == 'ppo-cartpole':
        ppo_cartpole_demo("Pendulum-v1")

    if request == 'ppo-cartpole-gs':
        grid_search(ppo_cartpole_demo, 
            epsilon=[0.2, 0.3, 0.4], 
            h=[16,32,64],
            #actor_lr=[1e-4, 2e-4, 3e-4],
            #critic_lr=[1e-3, 3e-4]
            )
