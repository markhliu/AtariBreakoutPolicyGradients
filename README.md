# AtariBreakoutPolicyGradients
![dig_tunnel](https://user-images.githubusercontent.com/50116107/132134465-a45448b8-120e-4928-8901-9d2ec6a54afb.gif)

Use policy gradients to train an agent to play the Atari Breakout game
Use policy gradients to play the Atari Breakout game
The goal is to train the agent so that it leans to dig a tunnel on the side
of the all to send the ball to the back to score more efficiently.

In 2013, the DeepMind team achieved this by using a different approach, namely,
the Deep Q Learning method. 

Here I am using policy gradients approach, inspired by this post:
http://karpathy.github.io/2016/05/31/rl/
by Stanford computer scientist Andrej Karpathy.

The code I used is largely based on Andrej's code for Atari Pong below
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

Andrej uses the Atari Pong game but I use Breakout here. 
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

This code below is for training only;
After about 250,000 episodes of traing, go to the pg_breakout_test.py to see 
the action, and if you want to capture how the agent digs a tunnel, follow the 
third script tunnel.py

you can see the action on my website below as well
https://gattonweb.uky.edu/faculty/lium/v/dig_tunnel.gif
 
