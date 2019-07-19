
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet_baseline(nn.Module):

    """
        A MLP with 2 hidden layer
        
        observation_dim (int): number of observation features
        action_dim (int): Dimension of each action
        seed (int): Random seed
    """

    def __init__(self, observation_dim, action_dim, seed):
        super(QNet_baseline, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, observations):
        """
           Forward propagation of neural network
           
        """

        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class QNet_3hidden(nn.Module):

    """
        A MLP with 3 hidden layer
        
        observation_dim (int): number of observation features
        action_dim (int): Dimension of each action
        seed (int): Random seed
    """

    def __init__(self, observation_dim, action_dim, seed):
        super(QNet_3hidden, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, observations):
        """
           Forward propagation of neural network
           
        """

        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    
class QNet_dropout(nn.Module):

    """
        A MLP with 2 hidden layer and dropout
        
        observation_dim (int): number of observation features
        action_dim (int): Dimension of each action
        seed (int): Random seed
    """

    def __init__(self, observation_dim, action_dim, seed):
        super(QNet_dropout, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_dim, 128)
        self.fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(64, action_dim)

    def forward(self, observations):
        """
           Forward propagation of neural network
           
        """

        x = F.relu(self.fc1(observations))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
class QNet_dueling(nn.Module):
    """
    Dueling DQN architecture. See: https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=128):

        super(QNet_dueling, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # value layer
        self.fc_v = nn.Linear(fc2_units, 1)
        # advantage layer
        self.fc_a = nn.Linear(fc2_units, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc_a(x)
        # Combine the value and advantage streams to final output.
        # Nomalized a with minus a.mean
        x = v + (a - a.mean(1).unsqueeze(1))
        return x