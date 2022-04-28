from util.structure import SparseMatrix
from collections import defaultdict


class Relation(object):
    def __init__(self, conf, relation=None):
        self.config = conf
        self.user = {}
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.relation_matrix = self.__generate_set()

    def __generate_set(self):
        triple = []
        for line in self.relation:
            user1, user2, weight = line
            # add relations to dict
            self.followees[user1][user2] = weight
            self.followers[user2][user1] = weight
            if user1 not in self.user:
                self.user[user1] = len(self.user)
            if user2 not in self.user:
                self.user[user2] = len(self.user)
            triple.append([self.user[user1], self.user[user2], weight])
        return SparseMatrix(triple)

    def row(self, u):
        # return user u's followees
        return self.relation_matrix.row(self.user[u])

    def col(self, u):
        # return user u's followers
        return self.relation_matrix.col(self.user[u])

    def elem(self, u1, u2):
        return self.relation_matrix.elem(u1, u2)

    def weight(self, u1, u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    def size(self):
        return self.relation_matrix.size

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
