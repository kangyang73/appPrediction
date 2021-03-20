import gym
import numpy as np
import environment
import model
import knn
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'AppPrediction-v4', '')
flags.DEFINE_integer('step_max', 2314, '')
flags.DEFINE_bool('random', False, '')
flags.DEFINE_integer('total_step', 300000, '')
flags.DEFINE_bool('exploration', False, '')
flags.DEFINE_float('epsilon', 0.1, '')
flags.DEFINE_string('actions_file', '../data/app/apps.npy', '')

class Senario:
  def main(self):
    self.train_st = 0
    self.env = environment.createPredictionEnv(gym.make(FLAGS.env))
    state_dim = self.env.observation_space.shape
    action_dim = self.env.action_space.shape
    action_k = knn.KNN(self.env, action_set=knn.load_action(FLAGS.actions_file)).get_action_k
    self.agent = model.DDPG(state_dim=state_dim, action_dim=action_dim, env_dtype=str(self.env.action_space.high.dtype))
    while self.train_st < FLAGS.total_step:
      self.exp(custom_policy=action_k)
      self.train_st += 1
    self.env.monitor.close()

  def exp(self, monitor=False, custom_policy=None):
    self.env.monitor.configure(monitor)
    state = self.env.reset()
    self.agent.reset(state)
    Reward = 0.
    step = 1
    term = False
    while not term:
      if FLAGS.random:
        exectute_action = self.env.action_space.sample()
        self.agent.action = exectute_action
        action_k = exectute_action
      else:
        if FLAGS.exploration and np.random.uniform() < FLAGS.epsilon:
          exectute_action = self.env.action_space.sample()
          action_k = exectute_action
        else:
          action = self.agent.act()
          Action_k = custom_policy(action)
          reward_k = Reward * np.ones(len(Action_k), dtype=self.env.action_space.high.dtype)
          term_k = np.zeros(len(Action_k), dtype=np.bool)
          max_qfunction_index = self.agent.action_k_policy(action, Action_k, reward_k, term_k)
          action_k = Action_k[max_qfunction_index]
          exectute_action = action_k
      state, reward, term, info = self.env.step(exectute_action)
      term = (step >= FLAGS.step_max) or term
      shape_reward = self.env.shape_reward(reward)
      self.agent.observe(shape_reward, term, state, action_k=action_k)
      Reward += reward
      step += 1
    self.env.render(mode='human')
    return Reward

if __name__ == '__main__':
  Senario().main()
