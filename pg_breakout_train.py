"""

The goal of this repo is to train the agent so that it learns to dig a tunnel on the side
of the all to send the ball to the back of the wall to score more efficiently.

In 2013, the DeepMind team achieved this by using a different approach, namely,
the Deep Q Learning method. Policy gradients is a different method in reinforcement learning.

Here I am using the policy gradients approach, inspired by this post:
http://karpathy.github.io/2016/05/31/rl/
by Stanford computer scientist Andrej Karpathy.

The code I used is largely based on Andrej's code for Atari Pong below
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
After about 250,000 episodes of training, go to the pg_breakout_test.py to see 
the action, and if you want to capture how the agent digs a tunnel, follow the 
third script tunnel.py

you can see the action in the gif above or on my website below as well
https://gattonweb.uky.edu/faculty/lium/v/dig_tunnel.gif
 
"""

# Install gym first by: pip install gym
import numpy as np
import pickle
import gym

# Hyperparameters
num_actions = 4
H = 200 
batch_size = 10 
learning_rate = 1e-4
gamma = 0.99 
decay_rate = 0.99 
D = 80 * 80 # input dimensionality: 80x80 grid

# Create model
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) 
model['W2'] = np.random.randn(num_actions,H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } 
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } 

# Define the one-hot encoder and softmax functions
def onehot_encoder(action):
    onehot=np.zeros((1,num_actions))
    onehot[0,action]=1
    return onehot

def softmax(x):
    xi=np.exp(x)
    return xi/xi.sum() 

# Preprocss the pixels
def prepro(I):
  I = I[35:195] 
  I = I[::2,::2,0] 
  I[I == 144] = 0 
  I[I == 109] = 0 
  I[I != 0] = 1 
  return I.astype(np.float).ravel()

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 
  logp = np.dot(model['W2'], h)
  p = softmax(logp)
  return p, h 

def policy_backward(eph, epdlogp):
  dW2 = np.dot(epdlogp.T, eph)
  dh = np.dot(epdlogp, model['W2'])
  dh[eph <= 0] = 0 
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Breakout-v0")
observation = env.reset()
prev_x = None 
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

# Count the number of lives to determine when to give reward -1
lives=5

while episode_number<250000:
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x
  
  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action=np.random.choice([0,1,2,3], p=aprob)
  # record various intermediates (needed later for backprop)
  xs.append(x) 
  hs.append(h) 
  y = onehot_encoder(action)
  # gradient for adjust weights, magic part
  dlogps.append(y - aprob) 

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  # Not in Andrej's script, my addition to solve the reward structure problem
  if (lives-info["ale.lives"])==1:
    lives -= 1
    reward=-1
  reward_sum += reward

  drs.append(reward) 

  if done==True:
    episode_number += 1
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory
    
    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    # modulate the gradient with advantage (PG magic happens right here.)
    epdlogp *= discounted_epr 
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] 

    # Reset lives to 5 after each episode
    lives=5
    fire=True
    reward_sum = 0    
    observation = env.reset() 
    prev_x = None        
    
    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] 
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) 
    
    # Show progress and save model
    if episode_number % 10 == 0: 
        print(f'this is episode {episode_number}')
    if episode_number % 100 == 0: 
        pickle.dump(model, open('v0breakpg.p', 'wb'))
    if episode_number % 10000 == 0: 
        pickle.dump(model, open(f'v0breakpg{episode_number}.p', 'wb'))
    

