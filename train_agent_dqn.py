import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent1 import Agent

"""
Set parameters, see parameters.py
"""
class Params:
    def __init__(self):

        # Output folders
        self.WEIGHTS_FOLDER = "./outputs/"
        self.DQN_WEIGHTS = None

        # Use GPU when available
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training Process
        self.N_EPISODES = 300  # max episodes
        self.MAX_T = 1200        # max steps per episode

        # Agent
        self.AGENT_SEED = 0  # random seed for agent
        self.BUFFER_SIZE = int(1e4)  # replay buffer size
        self.BATCH_SIZE = 128  # minibatch size
        self.GAMMA = 0.99  # discount factor
        # Network
        self.NN_SEED = 0      # random seed for Pytorch operations / networks
        self.LR = 0.001       # learning rate of the dqn
        self.FC1_UNITS = 64   # size of first hidden layer, dqn
        self.FC2_UNITS = 128  # size of second hidden layer, dqn

        # Ornstein-Uhlenbeck Process
        self.MU = 0.       # average
        self.THETA = 0.15  # drift
        self.SIGMA = 0.2   # volatility

params = Params() # instantiate the parameters
# load parameters
WEIGHTS_FOLDER = params.WEIGHTS_FOLDER
DQN_WEIGHTS = params.DQN_WEIGHTS
MU = params.MU
THETA = params.THETA
SIGMA = params.SIGMA
AGENT_SEED = params.AGENT_SEED
N_EPISODES = params.N_EPISODES
MAX_T = params.MAX_T

"""
Create Environment
"""
# select how many agents to use
env = UnityEnvironment(file_name='./reacher20/Reacher.exe', no_graphics=True)  # 20 agents
#env = UnityEnvironment(file_name='./reacher1/Reacher.exe', no_graphics=False)  # 1 agent

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment, select if you want to train or not
env_info = env.reset(train_mode=True)[brain_name]
# env_info = env.reset(train_mode=False, )[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

"""
Train the agent
"""

agent = Agent(num_agents, state_size, action_size, AGENT_SEED)

def dqn(n_episodes=300, max_t=1000, print_every=50):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]  # if you want to watch, set train_mode=False
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        
        for t in range(max_t):
            actions = agent.act(states, 0.1)            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            # Agent learns over New Step
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            #print(rewards)
            if any(dones):
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_dqn.pth')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.local.state_dict(), WEIGHTS_FOLDER + 'checkpoint_dqn.pth')
            break

    return scores


# train agent
scores = dqn(n_episodes=N_EPISODES, max_t=MAX_T)
env.close()

# plot scores
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
