import numpy as np
from data.data import Data
from collections import defaultdict


class Sequence(Data):
    def __init__(self, conf, training, test):
        super(Sequence, self).__init__(conf, training, test)
        self.item = {}
        self.id2item = {}
        self.seq = {}
        self.id2seq = {}
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.original_seq = self.__generate_set()
        self.raw_seq_num = len(self.seq)
        self.item_num = len(self.item)

    def __generate_set(self):
        original_sequences = []
        seq_index = len(self.seq)
        item_index = len(self.item) + 1  # 0 as placeholder

        for seq in self.training_data:
            seq_data = self.training_data[seq]
            if len(seq_data) < 2:
                continue

            if seq not in self.seq:
                self.seq[seq] = seq_index
                self.id2seq[seq_index] = seq
                seq_index += 1

            for item in seq_data:
                if item not in self.item:
                    self.item[item] = item_index
                    self.id2item[item_index] = item
                    item_index += 1

            original_sequences.append((seq, [self.item[item] for item in seq_data]))

        for seq in self.test_data:
            if seq in self.seq:
                first_item = self.test_data[seq][0]
                self.test_set[seq][first_item] = 1
                self.test_set_item.add(first_item)

        return original_sequences

    def get_item_id(self, i):
        return self.item.get(i)

    def get_seq_id(self, i):
        return self.seq.get(i)
