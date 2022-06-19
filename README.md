# Torch-Agents

In early development. Inspired by TF-Agents, I hope to make a similar library for PyTorch - simple enough for me to understand, and powerful enough to express all my crazy ideas.

Currently supported: (Double) Deep Q Networks and Open AI Gym environments.

## Results

DDQN Learning Pong after 13 hours (2500 games):

https://user-images.githubusercontent.com/10812888/172884861-843fb7b5-8823-4017-9042-25c5aff88438.mp4

DDQN Learning Breakout after 16 hours (10k games):

https://user-images.githubusercontent.com/10812888/173042669-0029ad6b-4c67-4661-81e0-3c6fa1f86211.mp4

DDQN Learning Space Invaders after 10k games:

https://user-images.githubusercontent.com/10812888/173040244-69920778-4644-4dec-9e49-bf617cd038cc.mp4


## Installation

First, install PyTorch with the latest CUDA version for your platform from [pytorch.org](https://pytorch.org/get-started/locally/).

Then install other required Python libraries with pip:

```
pip3 install tensorboard
pip3 install gym[classic_control]
pip3 install gym[atari]
pip3 install gym[accept-rom-license]
pip3 install gym[other]
pip3 install gym[box2d]
```
