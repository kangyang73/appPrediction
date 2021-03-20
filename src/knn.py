import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('create_index', True, '')
flags.DEFINE_integer('knn', 10, '')

class KNN(object):
    def __init__(self, env, action_set=None):
        self.__env = env
        self.__action = action_set
        self._sklearn()

    def _sklearn(self):
        if FLAGS.create_index:
            self.__algorithm = NearestNeighbors(algorithm='auto', leaf_size=30, metric='euclidean', n_jobs=-1)
            self.__algorithm.fit(self.__action)
            joblib.dump(self.__algorithm, 'knn.pkl')
        else:
            self.__algorithm = joblib.load('knn.pkl')

    def get_action_k(self, ac, k=None):
        neighbors = k if k is not None else FLAGS.knn
        _, results = self.__algorithm.kneighbors([ac], neighbors)
        results = results[0]
        return [self.__action[val] for val in results]

    @property
    def discrete_actions(self):
        return self.__action

def load_action(filename):
    return np.load(filename, "r", allow_pickle=False)
