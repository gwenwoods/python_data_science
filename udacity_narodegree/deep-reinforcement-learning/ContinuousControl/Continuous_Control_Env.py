import numpy as np
import random
import copy
from collections import namedtuple, deque

from Actor_Critic_Networks import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

import dill as pickle


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# Noise class

OU_NOISE_SIGMA = 0.1 # 0.2          # Ornstein-Uhlenbeck noise - Mean Reversion Speed
OU_NOISE_THETA = 0.08 #         # Ornstein-Uhlenbeck noise - Volatility

class Noise:
    # Use Ornstein-Uhlenbeck process.

    def __init__(self, size, mu=0.0, theta=OU_NOISE_THETA, sigma=OU_NOISE_SIGMA):
        # Initialize the noise process:
        #
        #    mu: long-running mean
        #    theta: mean reversion speed
        #    sigma: volatility
        
        self.state = None # This is the state of noise. The dimension is the number of actions
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        # Reset the noise to mean (mu)
        self.state = copy.copy(self.mu)

    def sample(self):
        # Update noise and return the noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

#------------------------------------------
# ReplayBuffer class

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128       # minibatch size

ExperienceData = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    # The memory buffer for storing the experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, memory_name):
        #self.action_size = action_size
        self.memory_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        #self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        if (memory_name != None):
            with open(memory_name, "rb") as fp:   # Unpickling
                self.memory_buffer = pickle.load(fp)

    def add(self, state, action, reward, next_state, done):
        # Add an experience to the memory_buffer.
        experience = ExperienceData(state, action, reward, next_state, done)
        self.memory_buffer.append(experience)

    def sample(self):
        # Sample a set (batch_size) of experiences from the memory_buffer. 
        experiences = random.sample(self.memory_buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory_buffer)
    
#-------------------------------------
# Agent class

GAMMA = 0.99                       # discount factor
TAU = 1e-3                         # for soft update of target network parameters
LEARNING_RATE_ACTOR = 1e-3 # 5e-4         # learning rate of the actor network
LEARNING_RATE_CRITIC = 1e-3 # 5e-4        # learning rate of the critic network
WEIGHT_DECAY = 0                   # L2 weight decay

EPSILON = 0.5  #1.0                # explore vs. exploit noise process added to act step
EPSILON_DECAY = 1e-6               # decay rate for noise process

LEARN_EVERY_TIMESTEP = 40
LEARN_REPEAT = 10

class DDPG_Agent():
    """Use deterministic policy gradient approact to learn from the environment."""

    def __init__(self, state_size, action_size, memory_name, random_seed):
        # Agent constructor.
        #    state_feature_num (int): dimension of each state
        #    action_num (int): dimension of each action
        #    memory_name (string) : name of the replay buffer to load
        #    random_seed (int): random seed
        self.state_feature_num = state_size
        self.action_num = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON

        # Double Actor Network (use local and target network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR)

        # Double Critic Network (use local and target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LEARNING_RATE_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = Noise(self.action_num)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, memory_name)

    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        #if len(self.memory) > BATCH_SIZE:
        #    experiences = self.memory.sample()
        #    self.learn(experiences, GAMMA)
        
        if len(self.memory) > BATCH_SIZE and t%LEARN_EVERY_TIMESTEP == 0 and t>0:
            if t== 1000:
                print("Epsilon = ", elf.epsilon)
            for x in range(LEARN_REPEAT):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        # Returns actions for given state as per current policy.
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        # Update policy and value parameters using given batch of experience tuples.
        #  Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        #   where:
        #    actor_target(state) -> action
        #    critic_target(state, action) -> Q-value
        # 
        # params:
        #    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        #    gamma (float): discount factor
        
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        #  Soft update model parameters.
        #   θ_target = τ*θ_local + (1 - τ)*θ_target
        # 
        #  params:
        #    local_model: PyTorch model (weights will be copied from)
        #    target_model: PyTorch model (weights will be copied to)
        #    tau (float): interpolation parameter
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

