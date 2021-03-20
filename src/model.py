import tensorflow as tf
import network
from buffer import Buffer
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('up', 5000, '')
flags.DEFINE_bool('up_q', True, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('iteration', 5, '')
learning_actor = 0.0001
learning_critic = 0.001
tau = 0.001
discount = 0.99
actor_l2 = 0.0
critic_l2 = 0.01
ou_theta = 0.15
buffer_size = 500000
threads = 8
ou_sigma = 0.2

class DDPG:
  start_train = False
  def __init__(self, state_dim, action_dim, cus_policy=True, env_dtype=tf.float32):
    action_dim = list(action_dim)
    state_dim = list(state_dim)
    self.cus_policy = cus_policy
    self.buffer = Buffer(buffer_size, state_dim, action_dim, dtype=np.__dict__[env_dtype])
    self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads, log_device_placement=False, allow_soft_placement=True))
    self.actor_theta = network.actor_theta(state_dim, action_dim)
    self.critic_theta = network.critic_theta(state_dim, action_dim)
    self.actor_theta_t, update_actor_t = exponential_moving_averages(self.actor_theta, tau)
    self.critic_theta_t, update_critic_t = exponential_moving_averages(self.critic_theta, tau)
    states = tf.placeholder(env_dtype, [None] + state_dim, "states")
    is_training = tf.placeholder(tf.bool, name="is_training")
    action_test = network.actor(states, self.actor_theta)
    noise_init = tf.zeros([1]+action_dim, dtype=env_dtype)
    noise_var = tf.Variable(noise_init)
    self.ou_reset = noise_var.assign(noise_init)
    noise = noise_var.assign_sub(ou_theta * noise_init - tf.random_normal(action_dim, stddev=ou_sigma, dtype=env_dtype))
    action_exploration = action_test + noise
    action_cont = tf.placeholder(env_dtype, [None] + action_dim, "action_cont_space")
    actions_k = tf.placeholder(env_dtype, [None] + action_dim, "knn_actions")
    reward_k = tf.placeholder(env_dtype, [1], "reward_k")
    term_k = tf.placeholder(tf.bool, [1], "term_k")
    qfunction_eval = network.critic(states, actions_k, self.critic_theta)
    action_k_policy = tf.stop_gradient(tf.where(term_k, reward_k, reward_k + discount * qfunction_eval))
    qfunction = network.critic(states, action_test, self.critic_theta)
    meanq = tf.reduce_mean(qfunction, 0)
    wd_p = tf.add_n([actor_l2 * tf.nn.l2_loss(var) for var in self.actor_theta])
    loss_p = -meanq + wd_p
    optim_p = tf.train.AdamOptimizer(learning_rate=learning_actor)
    grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.actor_theta)
    optimize_p = optim_p.apply_gradients(grads_and_vars_p)
    with tf.control_dependencies([optimize_p]):
      train_actor = tf.group(update_actor_t)
    action_train = tf.placeholder(env_dtype, [FLAGS.batch_size] + action_dim, "action_train")
    action_train_k = tf.placeholder(env_dtype, [FLAGS.batch_size] + action_dim, "action_train_k")
    reward = tf.placeholder(env_dtype, [FLAGS.batch_size], "reward")
    states2 = tf.placeholder(env_dtype, [FLAGS.batch_size] + state_dim, "states2")
    term2 = tf.placeholder(tf.bool, [FLAGS.batch_size], "term2")
    tensor_cond = tf.constant(self.cus_policy, dtype=tf.bool, name="is_cus_p")
    q_train = network.critic(states, action_train, self.critic_theta)
    act2 = network.actor(states2, theta=self.actor_theta_t)
    full_act_policy2 = tf.cond(tensor_cond, lambda: action_train_k, lambda: act2,)
    qfunction2 = network.critic(states2, full_act_policy2, theta=self.critic_theta_t)
    qfunction_target = tf.stop_gradient(tf.where(term2, reward, reward + discount*qfunction2))
    td_error = q_train - qfunction_target
    ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
    wd_q = tf.add_n([critic_l2 * tf.nn.l2_loss(var) for var in self.critic_theta])
    loss_q = ms_td_error + wd_q
    optim_q = tf.train.AdamOptimizer(learning_rate=learning_critic)
    grads_and_vars_q = optim_q.compute_gradients(loss_q, var_list=self.critic_theta)
    optimize_q = optim_q.apply_gradients(grads_and_vars_q)
    with tf.control_dependencies([optimize_q]):
      train_critic = tf.group(update_critic_t)
    with self.sess.as_default():
      self._action_test = Res([states, is_training], action_test)
      self._action_exploration = Res([states, is_training], action_exploration)
      self._reset = Res([], self.ou_reset)
      self._train_actor = Res([states, action_train, action_train_k, reward, states2, term2, is_training], [train_critic])
      self._train_critic = Res([states, is_training], [train_actor])
      self._train = Res([states, action_train, action_train_k, reward, states2, term2, is_training], [train_actor, train_critic])
      self._action_k_p = Res([states, action_cont, actions_k, reward_k, term_k, is_training], [action_k_policy])
    self.sess.run(tf.global_variables_initializer())
    self.sess.graph.finalize()
    self.t = 0

  def reset(self, states):
    self._reset()
    self.state = states

  def act(self):
    states = np.expand_dims(self.state, axis=0)
    action = self._action_exploration(states, True)
    self.action = np.atleast_1d(np.squeeze(action, axis=0))
    return self.action

  def action_k_policy(self, action_cont, actions_k, reward_k, term_k):
    states = np.expand_dims(self.state, axis=0)
    action_cont = np.expand_dims(action_cont, axis=0)
    i = 0
    qfunction_values = []
    for action_k in actions_k:
      action_k = np.expand_dims(action_k, axis=0)
      qfunction_values.append(self._action_k_p(states, action_cont, action_k, [reward_k[i]], [term_k[i]])[0])
      i += 1
    return np.argmax(qfunction_values)

  def observe(self, reward, term, states2, action_k=None):
    states1 = self.state
    self.state = states2
    self.t = self.t + 1
    self.buffer.enqueue(states1, term, self.action, action_k, reward)
    if self.t > FLAGS.up:
      self.train()
    elif FLAGS.up_q and self.buffer.n > 1000:
      states, act, act_k, reward, state2, term2, info = self.buffer.minibatch(buffer_size=FLAGS.batch_size)
      for i in xrange(FLAGS.iteration):
        self._train_actor(states, act, act_k, reward, state2, term2, True)

  def train(self):
    if not self.start_train:
        self.start_train = True
    states, act, act_k, reward, state2, term2, info = self.buffer.minibatch(buffer_size=FLAGS.batch_size)
    for i in xrange(FLAGS.iteration):
      self._train(states, act, act_k, reward, state2, term2, True,)

  def __del__(self):
    self.sess.close()


class Res:
  def __init__(self, inputs, outputs, session=None):
    self._inputs = inputs if type(inputs) == list else [inputs]
    self._outputs = outputs
    self._session = session or tf.get_default_session()

  def __call__(self, *args):
    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self._inputs[argpos]] = arg
    res = self._session.run(self._outputs, feeds)
    return res

def exponential_moving_averages(theta, tau=0.001):
  ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
  update = ema.apply(theta)
  averages = [ema.average(x) for x in theta]
  return averages, update
