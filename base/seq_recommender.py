from base.recommender import Recommender
from data.sequence import Sequence
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
from util.sampler import next_batch_sequence_for_test


class SequentialRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(SequentialRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Sequence(conf, training_set, test_set)
        self.bestPerformance = []
        self.max_len = int(self.config['max.len'])
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(SequentialRecommender, self).print_model_info()
        print(f'Training Set Size: (sequence number: {self.data.raw_seq_num}, item number: {self.data.item_num})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def predict(self, seq, pos, seq_len):
        return -1

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        for n, batch in enumerate(next_batch_sequence_for_test(self.data, self.batch_size, max_len=self.max_len)):
            seq, pos, seq_len = batch
            seq_start = n * self.batch_size
            seq_end = (n + 1) * self.batch_size
            seq_names = [seq_full[0] for seq_full in self.data.original_seq[seq_start:seq_end]]
            candidates = self.predict(seq, pos, seq_len)
            for name, res in zip(seq_names, candidates):
                ids, scores = find_k_largest(self.max_N, res)
                item_names = [self.data.id2item[iid] for iid in ids if iid != 0 and iid <= self.data.item_num]
                rec_list[name] = list(zip(item_names, scores))
            if n % 100 == 0:
                process_bar(n, self.data.raw_seq_num / self.batch_size)
        process_bar(self.data.raw_seq_num, self.data.raw_seq_num)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        return 0

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure
