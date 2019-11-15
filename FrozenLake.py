# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')
env.reset()



for episode in range(total_episodes):
    while t > max_steps:
        action = choose_action(state)
        state2, reward, done, info = env.step(action)
        learn(state, state2, reward, action)
        state = state2


# [State(S), Action(A)][reward[Q?]]
# ACTION : nord, sud, est ou ouest
# [S,A] -> QTable (Q)