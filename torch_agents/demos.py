import json
import sys
import torch

import agents
import memory
import networks
import explore
import environments

# Demo of Double DQN with priority replay memory playing CartPole
def cartpole_demo():
    env = environments.GymEnv(name="CartPole-v1", render_mode="human", valid_actions=[0,1])
    net = networks.FCNetwork([4, 512, 512, 2])
    mem = memory.PrioritizedReplayMemory(1e5)
    exp = explore.EpsilonGreedyStrategy(
            random_explore_frames=1000,
            initial_epsilon=1,
            final_epsilon=0.1,
            linear_epsilon_delta=1e-4,
            eval_epsilon=0.01
            )
    agent = agents.dqn('CartPole', env, net, mem, exp, lr = 0.0001)
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
            random_explore_actions=50000//4,
            initial_epsilon=1,
            final_epsilon=0.1,
            linear_epsilon_delta=9e-7,
            eval_epsilon=0.01
            )
    agent = agents.dqn('Breakout', env, net, mem, exp, lr = 0.00005,
            max_episodes=10000,
            action_repeat=4,
            update_freq=4,
            target_update_freq=10000,
            checkpoint_filename="breakout.pt"
    )
    agent.train()

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python demos.py <demo name>")
        exit()

    request = sys.argv[1].casefold()

    if request == 'cartpole':
        cartpole_demo()

    if request == 'breakout':
        breakout_demo()
    