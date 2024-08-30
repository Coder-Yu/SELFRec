from base.graph_recommender import GraphRecommender
import tensorflow as tf
from scipy.sparse import eye
import numpy as np
from base.tf_interface import TFGraphInterface
import os
from util.loss_tf import bpr_loss
from data.social import Relation
from data.augmentor import GraphAugmentor
from util.sampler import next_batch_pairwise

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Paper: Socially-aware self-supervised tri-training for recommendation. KDD'21

class SEPT(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)
        args = self.config['SEPT']
        self.n_layers = int(args['n_layer'])
        self.ss_rate = float(args['ss_rate'])
        self.drop_rate = float(args['drop_rate'])
        self.instance_cnt = int(args['ins_cnt'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)

    def print_model_info(self):
        super(SEPT, self).print_model_info()
        # # print social relation statistics
        print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
        print('=' * 80)

    def get_social_related_views(self, social_mat, interaction_mat):
        social_matrix = social_mat.dot(social_mat)
        social_matrix = social_matrix.multiply(social_mat) + eye(self.data.user_num, dtype=np.float32)
        sharing_matrix = interaction_mat.dot(interaction_mat.T)
        sharing_matrix = sharing_matrix.multiply(social_mat) + eye(self.data.user_num, dtype=np.float32)
        social_matrix = self.social_data.normalize_graph_mat(social_matrix)
        sharing_matrix = self.social_data.normalize_graph_mat(sharing_matrix)
        return [social_matrix, sharing_matrix]

    def _create_variable(self):
        self.sub_mat = {'adj_values_sub': tf.placeholder(tf.float32), 'adj_indices_sub': tf.placeholder(tf.int64),
                        'adj_shape_sub': tf.placeholder(tf.int64)}
        self.sub_mat['sub_mat'] = tf.SparseTensor(self.sub_mat['adj_indices_sub'], self.sub_mat['adj_values_sub'],
                                                  self.sub_mat['adj_shape_sub'])

    def encoder(self, emb, adj, n_layers):
        all_embs = [emb]
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            emb = tf.math.l2_normalize(emb, axis=1)
            all_embs.append(emb)
        all_embs = tf.reduce_sum(all_embs, axis=0)
        return tf.split(all_embs, [self.data.user_num, self.data.item_num], 0)

    def social_encoder(self, emb, adj, n_layers):
        all_embs = [emb]
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            emb = tf.math.l2_normalize(emb, axis=1)
            all_embs.append(emb)
        all_embs = tf.reduce_sum(all_embs, axis=0)
        return all_embs

    def build(self):
        super(SEPT, self).build()
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.data.user_num, self.emb_size]))
        self.item_embeddings = tf.Variable(initializer([self.data.item_num, self.emb_size]))
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self._create_variable()
        self.bi_social_mat = self.social_data.get_birectional_social_mat()
        social_mat, sharing_mat = self.get_social_related_views(self.bi_social_mat, self.data.interaction_mat)
        social_mat = TFGraphInterface.convert_sparse_mat_to_tensor(social_mat)
        sharing_mat = TFGraphInterface.convert_sparse_mat_to_tensor(sharing_mat)
        # initialize adjacency matrices
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        self.norm_adj = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj)

        # averaging the view-specific embeddings
        self.rec_user_embeddings, self.rec_item_embeddings = self.encoder(ego_embeddings, self.norm_adj, self.n_layers)
        self.aug_user_embeddings, self.aug_item_embeddings = self.encoder(ego_embeddings, self.sub_mat['sub_mat'],
                                                                          self.n_layers)
        self.sharing_view_embeddings = self.social_encoder(self.user_embeddings, sharing_mat, self.n_layers)
        self.friend_view_embeddings = self.social_encoder(self.user_embeddings, social_mat, self.n_layers)

        # embedding look-up
        self.batch_user_emb = tf.nn.embedding_lookup(self.rec_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.v_idx)
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.neg_idx)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

    def label_prediction(self, emb):
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        prob = tf.matmul(emb, aug_emb, transpose_b=True)
        # avoid self-sampling
        # diag = tf.diag_part(prob)
        # prob = tf.matrix_diag(-diag)+prob
        prob = tf.nn.softmax(prob)
        return prob

    def sampling(self, logits):
        return tf.math.top_k(logits, self.instance_cnt)[1]

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        pos_examples = self.sampling(positive)
        return pos_examples

    def neighbor_discrimination(self, positive, emb):
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=2)

        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        pos_emb = tf.nn.embedding_lookup(aug_emb, positive)
        emb2 = tf.reshape(emb, [-1, 1, self.emb_size])
        emb2 = tf.tile(emb2, [1, self.instance_cnt, 1])
        pos = score(emb2, pos_emb)
        ttl_score = tf.matmul(emb, aug_emb, transpose_a=False, transpose_b=True)
        pos_score = tf.reduce_sum(tf.exp(pos / 0.1), axis=1)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.1), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        return ssl_loss

    def train(self):
        # training the recommendation model
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.reg * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        # self-supervision prediction
        social_prediction = self.label_prediction(self.friend_view_embeddings)
        sharing_prediction = self.label_prediction(self.sharing_view_embeddings)
        rec_prediction = self.label_prediction(self.rec_user_embeddings)
        # find informative positive examples for each encoder
        self.f_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction)
        self.sh_pos = self.generate_pesudo_labels(social_prediction, rec_prediction)
        self.r_pos = self.generate_pesudo_labels(social_prediction, sharing_prediction)
        # neighbor-discrimination based contrastive learning
        self.neighbor_dis_loss = self.neighbor_discrimination(self.f_pos, self.friend_view_embeddings)
        self.neighbor_dis_loss += self.neighbor_discrimination(self.sh_pos, self.sharing_view_embeddings)
        self.neighbor_dis_loss += self.neighbor_discrimination(self.r_pos, self.rec_user_embeddings)
        # optimizer setting
        loss = rec_loss
        loss = loss + self.ss_rate * self.neighbor_dis_loss
        v1_opt = tf.train.AdamOptimizer(self.lRate)
        v1_op = v1_opt.minimize(rec_loss)
        v2_opt = tf.train.AdamOptimizer(self.lRate)
        v2_op = v2_opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            # joint learning
            if epoch > self.maxEpoch / 3:
                sub_mat = {}
                dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
                adj_mat1 = self.data.convert_to_laplacian_mat(dropped_mat)
                sub_mat['adj_indices_sub'], sub_mat['adj_values_sub'], sub_mat[
                    'adj_shape_sub'] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat1)

                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    feed_dict.update({
                        self.sub_mat['adj_values_sub']: sub_mat['adj_values_sub'],
                        self.sub_mat['adj_indices_sub']: sub_mat['adj_indices_sub'],
                        self.sub_mat['adj_shape_sub']: sub_mat['adj_shape_sub'],
                    })
                    _, l1, l3, = self.sess.run([v2_op, rec_loss, self.neighbor_dis_loss], feed_dict=feed_dict)
                    print('training:', epoch + 1, 'batch', n, 'rec loss:', l1, 'con_loss:', self.ss_rate * l3)
            else:
                # initialization with only recommendation task
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx, self.v_idx: i_idx, self.neg_idx: j_idx}
                    _, l1 = self.sess.run([v1_op, rec_loss], feed_dict=feed_dict)
                    print('training:', epoch + 1, 'batch', n, 'rec loss:', l1)
            self.U, self.V = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])
            self.fast_evaluation(epoch)
        self.U, self.V = self.best_user_emb, self.best_item_emb

    def save(self):
        self.best_user_emb, self.best_item_emb = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])

    def predict(self, u):
        u = self.data.get_user_id(u)
        return self.V.dot(self.U[u])
