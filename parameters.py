import torch


class Params:
    def __init__(self):

        # Output folders
        self.WEIGHTS_FOLDER = "./outputs/"
        self.CRITIC_WEIGHTS = None #torch.load(self.WEIGHTS_FOLDER + "checkpoint_critic67.pth")
        self.ACTOR_WEIGHTS = None #torch.load(self.WEIGHTS_FOLDER + "checkpoint_actor67.pth")

        # Use GPU when available
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training Process
        self.N_EPISODES = 10000  # max episodes
        self.MAX_T = 1200  # max steps per episode

        # Agent
        self.AGENT_SEED = 0  # random seed for agent
        self.BUFFER_SIZE = int(1e4)  # replay buffer size
        self.BATCH_SIZE = 128  # minibatch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.001  # interpolation parameter for soft update of target parameters
        self.WEIGHT_DECAY = 0  # L2 weight decay

        # Network
        self.NN_SEED = 0  # random seed for Pytorch operations / networks
        self.LR_ACTOR = 0.001  # learning rate of the actor
        self.FC1_UNITS_ACTOR = 64  # size of first hidden layer, actor
        self.FC2_UNITS_ACTOR = 128  # size of second hidden layer, actor

        self.LR_CRITIC = 0.001  # learning rate of the critic
        self.FC1_UNITS_CRITIC = 64  # size of first hidden layer, critic
        self.FC2_UNITS_CRITIC = 128  # size of second hidden layer, critic

        # Ornstein-Uhlenbeck Process
        self.MU = 0.  # average
        self.THETA = 0.15  # drift
        self.SIGMA = 0.2  # volatility


