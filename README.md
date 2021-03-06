# Use DQN to play Ataxx

## Disclaimer

The code gets its form from [the DQN tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), which is quite helpful. 

The net now is still very stupid. Coupling with other algorithms like MCTS may gives better result, but there's no warranty.

## Dependency

```
pytorch
numpy
```

## Structure

```
env.py: the environment of the Ataxx
    whatever

net.py: the network and training loop
```

## Network

The DQN accepts a batch of (state, action) and produce a reward, so it's an approximation of the total reward:

$$Q(s^{t_0}, a) = \Sigma_{t = t_0}^{\infty} \gamma^{t-t_0}R(s^{(t)}, a)$$

Where the R is the predefined reward function, and $$\gamma$$ the decaying factor(we don't care much about the far future).

So to train the network:

1. One should have some memory of (state, action)'s to update the network

2. One plays the game to generate samples, at each state, simply choose the action with the largest Q(with some randomness at first, but decays as time goes)

3. After each episode, one samples from the memory, and use the loss between the calculated expectation and the previous calculation to backprop to the parameters

## TO DO

* Write a torch tensor version of the environment

* Find out which gamma and reward function works better

* (IMPORTANT) discard state_action, use state only to reduce feature space
