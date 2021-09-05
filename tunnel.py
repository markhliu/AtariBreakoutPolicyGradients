# to install imageio, use: pip install imageio
import numpy as np
import pickle
import gym
import imageio

# Hyperparameters
num_actions = 4
D = 80 * 80 
model = pickle.load(open('v0breakpg250000.p', 'rb'))

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

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 
  logp = np.dot(model['W2'], h)
  p = softmax(logp)
  return p, h 

env = gym.make("Breakout-v0")
observation = env.reset()
prev_x = None 
running_reward = []
reward_sum = 0
episode_number = 0

frames=[]

while True:
  frames.append(env.render(mode='rgb_array'))
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  aprob, h = policy_forward(x)
  action=np.random.choice([0,1,2,3], p=aprob)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward  

  if done==True:
    episode_number += 1
    print(f'episode {episode_number} reward total was {reward_sum}')
    # usually, if episode reward is above 60, you have a tunnel
    if reward_sum>60:
        imageio.mimsave(f"frames{episode_number}.gif", frames, fps=60)
        break
    running_reward.append(reward_sum)
    reward_sum = 0    
    observation = env.reset() 
    prev_x = None        
    
print(f'running mean: {np.mean(running_reward)}')

