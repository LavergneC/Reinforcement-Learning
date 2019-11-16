# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

import gym
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf
import time

env = gym.make('FrozenLake-v0')
env.reset()

total_episodes = 5000
max_steps = 100
epsilon = 0.5
lr_rate = 0.81
gamma = 0.96
Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    #print(action)
    return action

def learn(state, state2, reward, action):
    old_value = Q[state, action]
    #print("Rew:",reward)
    if state == state2 or state2 in prev_states:
        reward = -10
    learned_value = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = (1 - lr_rate) * old_value +  lr_rate * learned_value

rews = []
for episode in range(total_episodes):
    state = env.reset()
    t=0
    prev_states = []
    while t < max_steps:
        #env.render()
        action = choose_action(state)
        state2, reward, done, info = env.step(action)
        learn(state, state2, reward, action)
        rews.append(reward)
        state = state2
        t += 1
        if done :
           # print("made in",t)
            break
    v = round((episode/total_episodes)*100,2)
    print(f'{v}%')
print(Q)
print(rews.count(1.0), len(rews))
np.save("maTable",Q)


# [State(S), Action(A)][reward[Q?]]
# ACTION : nord, sud, est ou ouest
# [S,A] -> QTable (Q)
# epsilon : taux d'exploration => Si exploitation : pif