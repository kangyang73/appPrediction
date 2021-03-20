import numpy as np
import gym

def createPredictionEnv(env):
  acsp = env.action_space
  obsp = env.observation_space
  env_type = type(env)

  class PredictionEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__)
      higher = obsp.high
      lower = obsp.low
      sc = higher-lower
      self.state_c = (higher+lower)/2.0
      self.state_sc = sc / 2.0
      self.reward_sc = 1.0
      self.reward_c = 0.0
      self.observation_space = gym.spaces.Box(self.shape_observation(obsp.low), self.shape_observation(obsp.high))
      self.action_space = gym.spaces.SimBox(-np.ones_like(acsp.high), np.ones_like(acsp.high), env=env, top_n=env.action_space.top_n)

    def shape_observation(self, obs):
      return (obs-self.state_c) / self.state_sc

    def shape_reward(self, reward):
      return self.reward_sc * reward + self.reward_c

    def step(self, action):
      action_f = np.clip(action, self.action_space.low, self.action_space.high)
      obs, reward, term, info = env_type.step(self, action_f)
      return obs, reward, term, info
  penv = PredictionEnv()
  return penv
