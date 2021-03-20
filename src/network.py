import tensorflow as tf
import numpy as np
default_type = tf.float64
number1_actor = 1000
number2_actor = 400
number1_critic = 400
number2_critic = 200

def init(shape,ini=None):
  ini = ini or shape[0]
  temp = 1/np.sqrt(ini)
  return tf.random_uniform(shape, minval=-temp, maxval=temp, dtype=default_type)

def critic_theta(state_dim,action_dim):
  state_dim = state_dim[0]
  action_dim = action_dim[0]
  with tf.variable_scope("critic_theta"):
    critic_theta = [tf.Variable(init([state_dim, number1_critic]), name='1w'),
            tf.Variable(init([number1_critic], state_dim), name='1b'),
            tf.Variable(init([number1_critic+action_dim, number2_critic]), name='2w'),
            tf.Variable(init([number2_critic], number1_critic+action_dim), name='2b'),
            tf.Variable(tf.random_uniform([number2_critic, 1], -3e-4, 3e-4, dtype=default_type), name='3w'),
            tf.Variable(tf.random_uniform([1], -3e-4, 3e-4, dtype=default_type), name='3b')]
    return critic_theta

def actor_theta(state_dim,action_dim):
  state_dim = state_dim[0]
  action_dim = action_dim[0]
  with tf.variable_scope("actor_theta"):
    actor_theta = [tf.Variable(init([state_dim, number1_actor]), name='1w'),
            tf.Variable(init([number1_actor], state_dim), name='1b'),
            tf.Variable(init([number1_actor, number2_actor]), name='2w'),
            tf.Variable(init([number2_actor], number1_actor), name='2b'),
            tf.Variable(tf.random_uniform([number2_actor, action_dim], -3e-3, 3e-3, dtype=default_type), name='3w'),
            tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3, dtype=default_type), name='3b')]
    return actor_theta

def critic(obs, act, theta, name="critic"):
  with tf.variable_op_scope([obs, act], name, name):
    h0 = tf.identity(obs, name='h0')
    h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
    h1a = tf.concat([h1, act], 1)
    h2 = tf.nn.relu(tf.matmul(h1a, theta[2]) + theta[3], name='h2')
    qs = tf.matmul(h2, theta[4]) + theta[5]
    qfunction = tf.squeeze(qs, [1], name='h3-q')
    return qfunction

  
def actor(obs,theta,name='actor'):
  with tf.variable_op_scope([obs], name, name):
    h0 = tf.identity(obs, name='h0')
    h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
    h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
    h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
    action = tf.nn.tanh(h3, name='h4-action')
    return action


