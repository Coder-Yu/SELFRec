import numpy as np
import heapq
from collections import defaultdict
import time
from base.graph_recommender import GraphRecommender

class ItemKNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ItemKNN, self).__init__(conf, training_set, test_set)
        self.topk = int(self.config['topK'])
        self.shrinkage = int(self.config['shrinkage'])
        self.item_sim = dict()  # item_name -> list of (similar_item_name, similarity)

    def _cosine_similarity(self, u_dict, v_dict):
        """
        Compute cosine similarity between two users, with shrinkage regularization.
        """
        common_items = set(u_dict.keys()) & set(v_dict.keys())
        n_common = len(common_items)
        if n_common == 0:
            return 0.0

        dot = sum(u_dict[i] * v_dict[i] for i in common_items)
        norm_u = np.sqrt(sum(r ** 2 for r in u_dict.values()))
        norm_v = np.sqrt(sum(r ** 2 for r in v_dict.values()))
        raw_sim = dot / (norm_u * norm_v + 1e-8)

        # shrinkage adjustment
        adjusted_sim = (n_common / (n_common + self.shrinkage)) * raw_sim
        return adjusted_sim

    def train(self):
        """
        For each item, compute top-K most similar items using cosine similarity.
        """
        print(f"[ItemKNN] Computing item-item similarity with top-{self.topk}...")
        start = time.time()
        all_itemnames = list(self.data.training_set_i.keys())

        for idx, i_name in enumerate(all_itemnames):
            i_users = self.data.training_set_i[i_name]
            sims = []
            for j_name in all_itemnames:
                if i_name == j_name:
                    continue
                j_users = self.data.training_set_i[j_name]
                sim = self._cosine_similarity(i_users, j_users)
                if sim > 0:
                    sims.append((sim, j_name))
            self.item_sim[i_name] = heapq.nlargest(self.topk, sims)

            if idx % 100 == 0:
                elapsed = time.time() - start
                print(f"  [Item {idx}/{len(all_itemnames)}] processed, elapsed: {elapsed:.1f}s")

        print(f"[ItemKNN] Similarity computation done in {time.time() - start:.2f}s.")

    def predict(self, u_name):
        """
        Predict scores for all items for user u.
        Output: numpy array of shape [item_num] indexed by item_id.
        """
        pred_scores = np.zeros(self.data.item_num)

        u_items = self.data.training_set_u[u_name]

        score_dict = defaultdict(float)
        sim_sum_dict = defaultdict(float)

        for item_name, rating in u_items.items():
            if item_name not in self.item_sim:
                continue
            for sim, j_name in self.item_sim[item_name]:
                item_id = self.data.item[j_name]
                score_dict[item_id] += sim * rating
                sim_sum_dict[item_id] += sim

        for item_id, score in score_dict.items():
            pred_scores[item_id] = score / (sim_sum_dict[item_id] + 1e-8)

        return pred_scores
