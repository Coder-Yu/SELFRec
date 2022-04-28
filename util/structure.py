import numpy as np


class SparseMatrix():
    def __init__(self,triple):
        self.matrix_user = {}
        self.matrix_item = {}
        for item in triple:
            if item[0] not in self.matrix_user:
                self.matrix_user[item[0]] = {}
            if item[1] not in self.matrix_item:
                self.matrix_item[item[1]] = {}
            self.matrix_user[item[0]][item[1]] = item[2]
            self.matrix_item[item[1]][item[0]] = item[2]
        self.elemNum = len(triple)
        self.size = len(self.matrix_user), len(self.matrix_item)

    def row(self,r):
        if r not in self.matrix_user:
            return {}
        else:
            return self.matrix_user[r]

    def col(self,c):
        if c not in self.matrix_item:
            return {}
        else:
            return self.matrix_item[c]

    def dense_row(self,r):
        if r not in self.matrix_user:
            return np.zeros((1,self.size[1]))
        else:
            array = np.zeros((1,self.size[1]))
            ind = list(self.matrix_user[r].keys())
            val = list(self.matrix_user[r].values())
            array[0][ind] = val
            return array

    def dense_col(self,c):
        if c not in self.matrix_item:
            return np.zeros((1,self.size[0]))
        else:
            array = np.zeros((1,self.size[0]))
            ind = list(self.matrix_item[c].keys())
            val = list(self.matrix_item[c].values())
            array[0][ind] = val
            return array

    def elem(self,r,c):
        if not self.contain(r,c):
            return 0
        return self.matrix_user[r][c]

    def contain(self,r,c):
        if r in self.matrix_user and c in self.matrix_user[r]:
            return True
        return False

    def elem_count(self):
        return self.elemNum

    def size(self):
        return self.size


