import tensorflow as tf
import numpy as np


class TFGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def create_joint_sparse_adj_tensor(adj):
        '''
        return a sparse tensor with the shape (user number + item number, user number + item number)
        '''
        row, col = adj.nonzero()
        indices = np.array(list(zip(row, col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
        return adj_tensor
