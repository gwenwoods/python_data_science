[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Project Description

The goal of this project is to train an agent to navigate a large, square world, and collect/avoid bananas in it. 

![Trained Agent][image1]

When the agent collects a yellow banana, it gets a reward of +1; on the other hand, when the agent collects a blue banana, it gets  a -1 reward. There, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Here are Unity details of the environment:

```
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

So our inputs contains 37 continous values (features) and outputs are 4 discrete actions representing moves (forward, backward, turn left, turn right). The environment is considered solved when agents reaches average score of 13.0 on 100 consecutive episodes.


### Instructions

QNetworks.py : Containing several different types of basic deep networks building blocks.
Banana_Env.py : Including the Double_DQN_Agent class and ReplayBuffer class.

### Getting Started

The Navigation.ipynb contains the workflow of training the agent:
    1. Use requirement.txt to install the requirements package for the project. 
    2. The codes which trains a pre-defined agents to solve the environment.
    
Please follow the notebook step by step.