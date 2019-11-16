import numpy as np
import gym
import random
import time
import os
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

# Action space : 4 -> nombre d'action possible (direction)
action_space_size = env.action_space.n

# Nombre d'états possible, un pour chaque case du jeu: 4x4=16
state_space_size = env.observation_space.n

# on fabrique une matrice dont les lignes contiennent les actions
# et les colonnes les états
# ex: Etat 1 (Case 00 1) : value action 1, value action 2, value action 3, value action 4
#     Etat 2 (Case 01) : ...

# Cette ligne fabrique une q-table de 0
q_table = np.zeros((state_space_size, action_space_size))

print("Q-Table initialized with 0s", q_table)

num_episode = 20000 # Nb de partie
max_steps_per_episode = 100 # si l'agent n'a pas fini en n étapes -> 0 pts

# alpha : à quelle points les nouvelles infos sont importantes 
# 1= ignorer les info d'avant
# 0=les Q values ne sont pas changé par les news values
learing_rate = 0.05 #alpha
discount_rate = 0.99 #gamma

# exploration|exploitation balance
# epsilon, 1 = 100% de chance d'explorer
# epsilon, 0.5 = 50% de chance d'explorer
exploration_rate = 1 #epsilon
max_exploration_rate = 1
min_exploration_rate = 0.02
exploration_decay_rate = 0.001 # avec le temps, on explore moins

rewards_all_ep = []

for episode in range(num_episode):
    state = env.reset() #Reset du jeu
    done = False
    rewards_current_ep = 0

    for step in range(max_steps_per_episode):
        #exploration|exploitation trade
        exploitation_random_generated = random.uniform(0, 1) #Random entre 0 et 1
        if exploitation_random_generated > exploration_rate:
            # exploitation
            action = np.argmax(q_table[state,:]) #Choix de la meilleur qValue pour notre state
        else:
            # exploration
            action = env.action_space.sample() #Action random
        #play
        new_state, reward, done, info = env.step(action)

        #Update Q-table
        # new Qvalue pour l'état de départ du step = oldQValue et NewQValeu
        q_table[state, action] = q_table[state, action] * (1 - learing_rate) + learing_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        rewards_current_ep += reward

        if done == True:
            # Si trou ou case G
            break
    
    # maj taux d'exploration, en foncton de l'épisode et d'une loi exponentielle
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    if not episode%1000:
        print("exploration_rate: ",round(exploration_rate*100,2),'%')
    rewards_all_ep.append(rewards_current_ep)

rewards_per_1000_ep = np.split(np.array(rewards_all_ep), num_episode/1000)
count = 1000

print("-------Taux de réusite-------\n")

for r in rewards_per_1000_ep:
    print(count,": ", str(round(sum(r/1000)*100,2)),'%')
    count += 1000

print("\n\n -------------- Q-Table------------\n")
print(q_table)
np.save("maTable",q_table)