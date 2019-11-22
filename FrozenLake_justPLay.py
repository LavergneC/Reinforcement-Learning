import gym
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf
import time

graph = False
total_episodes = 5000
max_steps = 1000000
nb_sucess = 0

Q = np.load("maTable.npy")

env = gym.make('Taxi-v3')

def choose_action(state):
    action = np.argmax(Q[state, :])
    return action

for episode in range(total_episodes):
    state = env.reset()
    t=0
    while t < max_steps:
        if graph: print("***** Step",t," *****" )
        if graph: env.render()
        action = choose_action(state)
        state2, reward, done, info = env.step(action)
        nb_sucess += reward
        state = state2
        t += 1
        if done :
            if graph:
                if reward > 0.5:
                    print("Gagné !")
                else:
                    print("Raté.")
                time.sleep(2.9)
            break
        if graph: time.sleep(0.3)
print("Success:", nb_sucess/total_episodes*100,"%")