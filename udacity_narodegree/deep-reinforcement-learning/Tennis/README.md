# Deep Reinforcement Learning: Collaboration and Competition


### Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

This project uses Tennis Unity environment. In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The dimension of the observation space is 24. Each of the rackets and ball consists of 8 variables corresponding to its position and velocity. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
 

### Solving the Environment


The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. 
- We then take the maximum of these 2 scores, which results in a single **score** for each episode.

The environment is considered solved, when the average (over 100 consecutive episodes) of those **scores** is at least +0.5.
 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

1. The environment can be downloaded from one of the links below for all operating systems:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    - _For AWS_: To train the agent on AWS (without [enabled virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  The agent can **not** be watched without a virtual screen, but can be trained.  (_To watch the agent, one can follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
    
2. Place the file in the project folder, and unzip (or decompress) the file. Make sure the path is correct when initilize the enviroment. 
    - env = UnityEnvironment(file_name='/your_path/data/Tennis_Linux_NoVis/Tennis')
    
3. The project uses Jupyter Notebook, and requires Unity ML-Agents, NumPy and PyTorch. In the `Tennis.ipynb`, please 
run the following command to install the needed packages:

 **!pip -q install ./python**

### Instructions

The Notebook includes two approaches for the project:

1. Agents are able to share their observations and actions.
2. Agents are NOT allowed to share their observations and actions.

Each approach defines the learning algorithm, network architecture, multi-agents structure and agents-training process.

Please follow the steps in `Tennis.ipynb` to train the agent.

