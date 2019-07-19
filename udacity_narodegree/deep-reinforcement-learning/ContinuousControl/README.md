# Deep Reinforcement Learning: Continuous_Control


### Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

This project uses Reacher Unity environment. In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Distributed Training

The task is episodic. For this project, we will use the Unity environment that contains 20 identical agents, each with its own copy of the environment.  
 

### Solving the Environment

The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,

 - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
 - This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
2. Place the file in the project folder, and unzip (or decompress) the file. Make sure the path is correct when initilize the enviroment. 
    - env = UnityEnvironment(file_name='/your_path/data/Reacher_Linux_NoVis/Reacher.x86_64')
    
3. The project uses Jupyter Notebook, and requires Unity ML-Agents, NumPy and PyTorch. In the `Continuous_Control.ipynb`, please 
run the following command to install the needed packages:

 **!pip -q install ./python**

### Instructions

Please follow the steps in `Continuous_Control.ipynb` to train the agent. 

The Notebook uses the following module files:
- Continuous_Control_Env.py - the Noise class, ReplayBuffer class, DDPG_Agent class
- Actor_Critic_Networks.py - the Actor and Critic models