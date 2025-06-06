import numpy as np
import heapq
from collections import defaultdict
import time
from base.graph_recommender import GraphRecommender

class UserKNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(UserKNN, self).__init__(conf, training_set, test_set)
        self.topk = int(self.config['topK'])
        self.shrinkage = int(self.config['shrinkage'])
        self.user_sim = dict()  # user_id -> list of (neighbor_user, similarity)

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
        For each user, compute top-K most similar users using cosine similarity.
        Use usernames and item names throughout.
        """
        print(f"[UserKNN] Computing user-user similarity with top-{self.topk}...")
        start = time.time()
        all_usernames = list(self.data.training_set_u.keys())

        for idx, u_name in enumerate(all_usernames):
            u_items = self.data.training_set_u[u_name]
            sims = []
            for v_name in all_usernames:
                if u_name == v_name:
                    continue
                v_items = self.data.training_set_u[v_name]
                sim = self._cosine_similarity(u_items, v_items)
                if sim > 0:
                    sims.append((sim, v_name))
            self.user_sim[u_name] = heapq.nlargest(self.topk, sims)

            if idx % 100 == 0:
                elapsed = time.time() - start
                print(f"  [User {idx}/{len(all_usernames)}] processed, elapsed: {elapsed:.1f}s")

        print(f"[UserKNN] Similarity computation done in {time.time() - start:.2f}s.")

    def predict(self, u):
        """
        Predict scores for all items for user u.
        Output: numpy array of shape [item_num] indexed by item_id.       """


        pred_scores = np.zeros(self.data.item_num)

        score_dict = defaultdict(float)
        sim_sum_dict = defaultdict(float)

        for sim, v_name in self.user_sim[u]:
            for item_name, rating in self.data.training_set_u[v_name].items():

                item_id = self.data.item[item_name]
                score_dict[item_id] += sim * rating
                sim_sum_dict[item_id] += sim

        for item_id, score in score_dict.items():
            pred_scores[item_id] = score / (sim_sum_dict[item_id] + 1e-8)

        return pred_scores

