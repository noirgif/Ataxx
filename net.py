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
import sys
import os

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


play = env.Play()


def get_env():
    cb = play.b
    cb = torch.from_numpy(cb)
    # add two dimensions `channel` and `batch`
    return cb.type(LongTensor).unsqueeze(0).unsqueeze(0)


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
    """Deep Q-Net(not really)"""

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
GAMMA = -0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
memory = ReplayMemory(2048)

steps_done = 0


def select_action(state):
    """ select an action with given state
        state: LongTensor(cuda if applicable)
        NOTICE! it returns a numpy array for ease of calculation"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # is iterating through all samples a bad idea?
    moves = [i for i in env.get_moves(state.squeeze().squeeze().cpu().numpy())]
    if sample < eps_threshold:
        return random.choice(moves)
    # optimal move
    moves_ts = map(lambda x: torch.from_numpy(x).type(
        LongTensor).unsqueeze(0).unsqueeze(0), moves)
    moves_ts = map(lambda x: torch.cat([x, state], 1), moves_ts)
    state_action = torch.cat(list(moves_ts))
    x = model(
        Variable(state_action, volatile=True).type(FloatTensor))
    return moves[x.max(0)[1].data.sum()]


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(
        tuple(map(lambda s: s is not None, batch.next_state)))

    state_batch = Variable(torch.cat(batch.state).type(FloatTensor))
    action_batch = Variable(torch.cat(batch.action).type(FloatTensor))
    reward_batch = Variable(torch.cat(batch.reward))

    # no grad for next states
    non_final_next_states = [s for s in batch.next_state
                             if s is not None]

    # calculate next_state_values[non_final_mask]
    non_final_next_state_values = []
    for s in non_final_next_states:
        # convert to numpy array SIZExSIZE
        state = s.squeeze().squeeze().cpu().numpy()
        moves = [move for move in env.get_moves(state)]
        moves = map(lambda x: torch.from_numpy(x).type(
            LongTensor).unsqueeze(0).unsqueeze(0), moves)
        # concat with state to make input
        moves = map(lambda x: torch.cat([x, s], 1), moves)

        state_action = torch.cat(list(moves))
        state_action = Variable(state_action.type(FloatTensor), volatile=True)
        result = model(state_action)
        state_action.volatile = False
        non_final_next_state_values.append(result.max(0)[0].data)
    non_final_next_state_values = torch.cat(non_final_next_state_values)

    state_action_values = model(torch.cat([state_batch, action_batch], 1))

    next_state_values = Variable(reward_batch.data)
    next_state_values[non_final_mask] = non_final_next_state_values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # complute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    start_episode = 0
    num_episodes = 500

    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]):
            print("=> loading checkpoint '{}'".format(sys.argv[1]))
            checkpoint = torch.load(sys.argv[1])
            start_episode = checkpoint['episode']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            memory = checkpoint['memory']
            steps_done = checkpoint['steps_done']
            print("=> loaded episode {}"
                  .format(checkpoint['episode']))
        else:
            print("=> no checkpoint found at '{}'".format(sys.argv[1]))

    if len(sys.argv) > 2 and sys.argv[2] == 'play':
        play.reset()
        done = False
        while not done:
            # NO check!
            print(play.b)
            print("Enter your move")
            move = list(map(int, input().split(' ')))
            _, change = env.put(play.b, move)
            _, done = play.step(change)
            print(-play.b)
            if done:
                break
            # would be better without the random choice
            _, done = play.step(select_action(torch.from_numpy(
                play.b).type(LongTensor).unsqueeze(0).unsqueeze(0)))
        sys.exit(0)
    # instantiate a play
    for i_episode in range(start_episode, num_episodes):
        print("Episode {}".format(i_episode))
        play.reset()
        # state: SIZExSIZE array
        state = get_env()
        for t in count():
            action = select_action(state)
            reward, done = play.step(action)
            if t % 30 == 0:
                print("Turn {}:\n".format(t), play.b)
            reward = Tensor([reward])

            if not done:
                next_state = get_env()
            else:
                next_state = None
            # push a torch tensor rather than a numpy array
            action = torch.from_numpy(action).type(
                LongTensor).unsqueeze(0).unsqueeze(0)
            assert isinstance(action, LongTensor)
            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                break
        save_checkpoint({
            'episode': i_episode + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'memory': memory,
            'steps_done': steps_done
        })
