import json
import sys
import torch

import agents
import memory
import networks
import explore
import environments
import schedule

# Demo of Double DQN with priority replay memory playing CartPole
def cartpole_demo():
    replay_start_size = 1000

    env = environments.GymEnv(name="CartPole-v1", render_mode="human", valid_actions=[0,1])
    net = networks.FCNetwork([4, 32, 32, 2])
    mem = memory.PrioritizedReplayMemory(1e6)
    exp = explore.EpsilonGreedyStrategy(
        epsilon=schedule.Sequence([
            schedule.Flat(replay_start_size, 1.0),
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
        replay_start_size=replay_start_size,
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
    