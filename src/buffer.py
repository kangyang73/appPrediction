import numpy as np
import random

class Buffer:
  def __init__(self, buffer_size, state_dim, action_dim, dtype=np.float32):
    self.buffer_size = buffer_size
    state = np.concatenate(np.atleast_1d(buffer_size, state_dim), axis=0)
    action = np.concatenate(np.atleast_1d(buffer_size, action_dim), axis=0)
    self.observations = np.empty(state, dtype=dtype)
    self.actions = np.empty(action, dtype=np.float32)
    self.k_actions = np.empty(action, dtype=np.float32)
    self.rewards = np.empty(buffer_size, dtype=np.float32)
    self.terminals = np.empty(buffer_size, dtype=np.bool)
    self.info = np.empty(buffer_size, dtype=object)
    self.n = 0
    self.i = -1

  def reset(self):
    self.n = 0
    self.i = -1

  def enqueue(self, observation, terminal, action, k_action, reward, info=None):
    self.i = (self.i + 1) % self.buffer_size
    self.observations[self.i, ...] = observation
    self.terminals[self.i] = terminal
    self.actions[self.i, ...] = action
    self.k_actions[self.i, ...] = k_action
    self.rewards[self.i] = reward
    self.info[self.i, ...] = info
    self.n = min(self.buffer_size-1, self.n + 1)
  
  def minibatch(self, buffer_size):
    indices = np.zeros(buffer_size, dtype=np.int)
    for k in range(buffer_size):
      invalid = True
      i = 0
      while invalid:
        i = random.randint(0, self.n-2)
        if i != self.i and not self.terminals[i]:
          invalid = False
      indices[k] = i
    state = self.observations[indices, ...]
    action = self.actions[indices]
    k_action = self.k_actions[indices]
    rew = self.rewards[indices]
    state2 = self.observations[indices+1, ...]
    term = self.terminals[indices+1]
    info = self.info[indices, ...]
    return state, action, k_action, rew, state2, term, info
    
  def __repr__(self):
    indices = range(0, self.n)
    state = self.observations[indices, ...]
    action = self.actions[indices]
    k_action = self.k_actions[indices]
    rew = self.rewards[indices]
    term = self.terminals[indices]
    string = """
    OBSERVATIONS
    {}
    ACTIONS
    {}
    K ACTIONS
    {}
    REWARDS
    {}
    TERMINALS
    {}
    """.format(state, action, k_action, rew, term)
    return string
