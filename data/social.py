from collections import defaultdict
from data.graph import Graph
import numpy as np
import scipy.sparse as sp


class Relation(Graph):
    def __init__(self, conf, relation, user):
        super().__init__()
        self.config = conf
        self.social_user = {}
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.user = user
        self.__initialize()

    def __initialize(self):
        idx = []
        for n, pair in enumerate(self.relation):
            if pair[0] not in self.user or pair[1] not in self.user:
                idx.append(n)
        for item in reversed(idx):
            del self.relation[item]
        for line in self.relation:
            user1, user2, weight = line
            # add relations to dict
            self.followees[user1][user2] = weight
            self.followers[user2][user1] = weight

    def get_social_mat(self):
        row, col, entries = [], [], []
        for pair in self.relation:
            row += [self.user[pair[0]]]
            col += [self.user[pair[1]]]
            entries += [1.0]
        social_mat = sp.csr_matrix((entries, (row, col)), shape=(len(self.user), len(self.user)), dtype=np.float32)
        return social_mat

    def get_birectional_social_mat(self):
        social_mat = self.get_social_mat()
        bi_social_mat = social_mat.multiply(social_mat)
        return bi_social_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        (row_np_keep, col_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (row_np_keep, col_np_keep)), shape=adj_shape, dtype=np.float32)
        return self.normalize_graph_mat(tmp_adj)

    def weight(self, u1, u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    def get_followers(self, u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def get_followees(self, u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def has_followee(self, u1, u2):
        if u1 in self.followees:
            if u2 in self.followees[u1]:
                return True
            else:
                return False
        return False

    def has_follower(self, u1, u2):
        if u1 in self.followers:
            if u2 in self.followers[u1]:
                return True
            else:
                return False
        return False

    def size(self):
        return len(self.followers), len(self.relation)
