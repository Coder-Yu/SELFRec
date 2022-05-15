import numpy as np
from data.data import Data


class Sequence(Data):
    def __init__(self, conf, training, test):
        super(Sequence, self).__init__(conf, training, test)
        self.item = {}
        self.id2item = {}
        self.__generate_set()
        self.raw_seq_num = len(self.training_data)
        self.item_num = len(self.item)

    def __generate_set(self):
        for seq in self.training_data:
            for item in seq:
                if item not in self.item:
                    self.item[item] = len(self.item)
                    self.id2item[self.item[item]] = item



    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]





