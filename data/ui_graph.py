import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for user, item, rating in self.training_data:
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        for user, item, rating in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = rating
                self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
