# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:35:51 2022

@author: user
"""

import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class Params:
    def __init__(self):

        # Output folders
        self.WEIGHTS_FOLDER = "./outputs/"

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


class DQN(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        """
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)
        """
        
    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""       
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
    
class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out))
        return out    


"""
Set parameters,
"""
params = Params()  # instantiate parameters
BUFFER_SIZE = params.BUFFER_SIZE
BATCH_SIZE = params.BATCH_SIZE
GAMMA = params.GAMMA
LR = params.LR
FC1_UNITS = params.FC1_UNITS
FC2_UNITS = params.FC2_UNITS
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network
device = params.DEVICE
print(device)        


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # DQN Network 
        self.local = DQN(state_size, action_size, random_seed, FC1_UNITS, FC2_UNITS).to(device)
        self.target = DQN(state_size, action_size, random_seed, FC1_UNITS, FC2_UNITS).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=LR) 
        print(self.local)
        print(self.target)


        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward for all agents
        for state, action, reward, next_state, done in zip(state, action, reward, next_state, done):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()
        if random.random() > eps:
            return action
        else:
            return np.random.randint(4,size=(action.shape[0], action.shape[1]))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update dqn ---------------------------- #
        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets_next = self.target(next_states).detach().max(1)[0].unsqueeze(1) #(None, 1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.local(states).gather(1, actions.detach().max(1)[0].unsqueeze(1))
        #print(actions.detach().max(1)[0].unsqueeze(1))
        #print(Q_expected.shape)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local, self.target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)




class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
