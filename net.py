"""Import packages
    torch.nn : neural networks
    torch.nn.functional : functions
"""
import math
import random
from collections import namedtuple
from copy import deepcopy
from itertools import count
import numpy as np
import env


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# use gpu
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def get_env():
    cb = env.get()
    cb = torch.from_numpy(cb)
    # add two dimensions `channel` and `batch`
    return cb.unsqueeze(0).unsqueeze(0).type(Tensor)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q-Net"""

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(1568, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


BATCH_SIZE = 128
GAMMA = -0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
memory = ReplayMemory(1000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        optim = None
        reward = -25252
        for i in env.get_moves():
            state_action = torch.cat([state, torch.from_numpy(i)]).unsqueeze(0)
            x = model(
                Variable(state_action, volatile=True).type(FloatTensor))
            if x > reward:
                optim = i
                reward = x
        return optim    
    else:
        return LongTensor([random.randrange(7) for i in range(4)])


episode_durations = []


last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    print(batch.action)

    non_final_mask = ByteTensor(
        tuple(map(lambda s: s is not None, batch.next_state)))

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    values = []

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    for s in batch.next_state:
        for move in env.get_moves(batch.next_state):


    
    # no grad for next states
    non_final_next_states = Variable(torch.cat(),
                                        volatile=True)
    # calculate next_state_values[non_final_mask]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    state_action_values = model(torch.cat([state_batch, action_batch], 1))

    # clean the volatile flag, ends up with require_grad = false only
    next_state_values.volatile = False

    

    # complute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == '__main__':
    num_episodes = 2000
    for i_episode in range(num_episodes):
        env.reset()
        state = get_env()
        print("State: ", state)
        for t in count():
            action = select_action(state)
            print(action)
            reward, done = env.step(action)
            reward = Tensor([reward])

            if not done:
                next_state = get_env()
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                episode_durations.append(t + 1)
                break
