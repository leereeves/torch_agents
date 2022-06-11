# DeepQ

A PyTorch implementation of Deep Q Learning (Minh 2015), with the following improvements:

* Double Q Learning to reduce overestimation (Van Hasselt 2016)

* Prioritized Experience Replay for faster learning (Schaul 2016)

* Combined Experience Replay (Zhang 2017)

## Results

Pong after 13 hours (2500 games):

https://user-images.githubusercontent.com/10812888/172884861-843fb7b5-8823-4017-9042-25c5aff88438.mp4

Breakout after 16 hours (10k games):

https://user-images.githubusercontent.com/10812888/173042669-0029ad6b-4c67-4661-81e0-3c6fa1f86211.mp4

Space Invaders after 10k games:

https://user-images.githubusercontent.com/10812888/173040244-69920778-4644-4dec-9e49-bf617cd038cc.mp4


## Installation

## References

Brockman, Greg, et al. "Openai gym." arXiv preprint arXiv:1606.01540 (2016).

https://arxiv.org/pdf/1606.01540.pdf

https://github.com/openai/gym

https://www.gymlibrary.ml/

He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.

https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf

Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.

https://daiwk.github.io/assets/dqn.pdf

Schaul, Tom, et al. "Prioritized experience replay."  ICLR 2016 (2016).

https://arxiv.org/pdf/1511.05952.pdf

Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.

https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154

Zhang, Shangtong, and Richard S. Sutton. "A deeper look at experience replay." arXiv preprint arXiv:1712.01275 (2017).

https://arxiv.org/pdf/1712.01275.pdf
