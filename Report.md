[//]: # (Image References)

[image1]: scores.png "Scores for all 20 agents"
[image2]: scores_mean.png "Mean scores"


### Introduction

This report describes the implementation of the Deep Deterministic Poligy Gradient (DDPG) method to solve the Unity Reacher environment. More information on DDPG can be found in the [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

### Implementation

The agent was first trained using small adaptions to the example code provided as part of Udacity's course with the hyperparameters set to those in the DDPG paper. However, this resulted in an agent that could not train very succesfully, so the hyperparameters were changed one by one to examine the impact on training. It took many attempts to find a set of hyperparameters and a network architecture that could succesfully train the agent to solve the environment. Many combinations either started to train and then plateaud, or dropped off towards zero reward.

Eventually, the hyperparameters below were found to solve the environment:
* BUFFER_SIZE = int(1e6)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.95            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor 
* LR_CRITIC = 1e-3        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay
* UPDATE_EVERY = 20       # how often to update the network
* EPSILON = 0.5           # how much to scale the noise

The change with the most impact was increasing the number of observations per episode (max_t) from 700 to 1000. This gives the agent a greater reward signal, as well as more states to observe.

A smaller network size proved to train more succesfully. The actor and crtic models are defined in model.py and both have 3 fully connected layers. A batch normalisation follows the first layer, which improves training when the state inputs are scaled very differently. The unit sizes are 256 for the first layer and 128 units for the second layer for both networks. In the critic model, the action is concatenated after the first dense layer and batch normalisation. The weights are initialised as recommended by the paper, aside from a difference in scale between the final layers. The actor final layer is initialised with uniform noise between -3e-3 and 3e-3, whereas the critic is initialised with uniform noise between -3e-4 and 3e-4.

EPSILON has been added to scale the Ornstein-Uhlenbeck noise and reduce the amount of random search in the action. This is decayed by multiplying by 0.995 for every batch of learning.

Following guidance, the network is not updated at every step, in an attempt to prevent oscillations in the training. It is updated 10 times every 20 steps. The number of agents is not actually that important in this implementation. Each agent collects a sequence of observations and records it in the shared memory buffer. These are then randomly sampled by the algorithm to perform updates. However, multiple agents are advantageous in providing a large number of uncorrelated experiences.

### Results

The plots below show the training score for all agents and the average score across all agents for each episode in training. The environment was solved in 108 episodes.

![Scores][image1]
![Average Scores][image1]

By keeping all hyperparameter fixed and only changing max_t from 1000 to 700, training plateaus around an average score of 7. This highlights the important of a strong reward signal, which many require a larger number of observations.

Without batch normalisation and with an epsilon value of 1, the agent plateaus around an average score of 20.

### Potential Future Work
Many hyperparameter combinations were investigated before max_t was increased and found to have differing impacts on the training. These could be further examined and recorded to determine a set of successful hyperparameter combinations and provide guidance on which are most important to focus on when tuning for new problems.

An alternative approach to this problem would be to use [Proximal Policy Optimisation (PPO)](https://arxiv.org/pdf/1707.06347.pdf). 

Distributed algorithms, such as [Distributed Distributional Deep Deterministic Policy Gradient (D4PG)](https://openreview.net/pdf?id=SyZipzbCb) may possibly improve performance.


