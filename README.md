# Atari Breakout Policy Gradients
![dig_tunnel](https://user-images.githubusercontent.com/50116107/132134465-a45448b8-120e-4928-8901-9d2ec6a54afb.gif)   ![breakout2](https://user-images.githubusercontent.com/50116107/132328577-256d3026-aea9-4897-8d59-b4886308ba65.gif)


Use policy gradients to train an agent to play the Atari Breakout game.


The goal of this repo is to train the agent so that it learns to dig a tunnel on the side
of the wall to send the ball to the back of the wall to score more efficiently.

In 2013, the DeepMind team achieved this by using a different approach, namely,
the Deep Q Learning method. Policy Gradient is a different method in reinforcement learning.

Here I am using the policy gradients approach, inspired by this post:
http://karpathy.github.io/2016/05/31/rl/
by Stanford computer scientist Andrej Karpathy.

The code I used is largely based on Andrej's code for the Atari Pong below
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

Andrej uses the Atari Pong game but I use the Breakout game here. 
With Breakout, things are more complicated in two ways: 
first, in the Pong game, you only need to move the 
paddle up or down, which is a binary choice. Second, in the Pong game,
the reward structure is more nuanced: -1 for missing,
1 for winning, and 0 if nothing happens. 

In contrast, in Breakout, you need four choices: left, right, no movement, 
or firing the ball. Second, the reward is either 0 or 1. There is no reward
of -1 if the paddle misses the ball.

I changed the binary-classification to multiple-classification; 
Then I hardcode in a reward of -1 if the paddle misses the ball by counting 
the numbers of lives left for the agent. 

This script pg_breakout_train.py is for training only.
After about 250,000 episodes of training, go to pg_breakout_test.py to see 
the action, and if you want to capture how the agent digs a tunnel, follow the 
third script tunnel.py

you can see the action in the gif above or on my website below as well
https://gattonweb.uky.edu/faculty/lium/v/dig_tunnel.gif

https://gattonweb.uky.edu/faculty/lium/v/breakout2.gif
 
Enjoy!

Mark
