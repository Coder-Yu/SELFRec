from base.graph_recommender import GraphRecommender
import tensorflow as tf
from util.conf import OptionConf
from util.loss import bpr_loss,infoNCE
from util.sampler import next_batch_pairwise
from base.tf_interface import TFGraphInterface
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCL, self).__init__(conf, training_set, test_set)
        self.performance = []
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])

    def LightGCN_encoder(self,emb,adj,n_layers):
        all_embs = []
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            all_embs.append(emb)
        all_embs = tf.reduce_mean(all_embs, axis=0)
        return tf.split(all_embs, [self.data.user_num, self.data.item_num], 0)

    def perturbed_LightGCN_encoder(self,emb,adj,n_layers):
        all_embs = []
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            random_noise = tf.random.uniform(emb.shape)
            emb += tf.multiply(tf.sign(emb),tf.nn.l2_normalize(random_noise, 1)) * self.eps
            all_embs.append(emb)
        all_embs = tf.reduce_mean(all_embs, axis=0)
        return tf.split(all_embs, [self.data.user_num, self.data.item_num], 0)

    def build(self):
        super(SimGCL, self).build()
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.data.user_num, self.emb_size]))
        self.item_embeddings = tf.Variable(initializer([self.data.item_num, self.emb_size]))
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        #adjaceny matrix
        self.norm_adj = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj)
        #encoding
        self.main_user_embeddings, self.main_item_embeddings = self.LightGCN_encoder(ego_embeddings,self.norm_adj,self.n_layers)
        self.perturbed_user_embeddings1, self.perturbed_item_embeddings1 = self.perturbed_LightGCN_encoder(ego_embeddings,self.norm_adj, self.n_layers)
        self.perturbed_user_embeddings2, self.perturbed_item_embeddings2 = self.perturbed_LightGCN_encoder(ego_embeddings, self.norm_adj, self.n_layers)
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

    def save(self):
        self.best_user_emb, self.best_item_emb = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def calc_cl_loss(self):
        p_user_emb1 = tf.nn.embedding_lookup(self.perturbed_user_embeddings1, tf.unique(self.u_idx)[0])
        p_item_emb1 = tf.nn.embedding_lookup(self.perturbed_item_embeddings1, tf.unique(self.v_idx)[0])
        p_user_emb2 = tf.nn.embedding_lookup(self.perturbed_user_embeddings2, tf.unique(self.u_idx)[0])
        p_item_emb2 = tf.nn.embedding_lookup(self.perturbed_item_embeddings2, tf.unique(self.v_idx)[0])
        normalize_emb_user1 = tf.nn.l2_normalize(p_user_emb1, 1)
        normalize_emb_user2 = tf.nn.l2_normalize(p_user_emb2, 1)
        normalize_emb_item1 = tf.nn.l2_normalize(p_item_emb1, 1)
        normalize_emb_item2 = tf.nn.l2_normalize(p_item_emb2, 1)
        cl_loss = infoNCE(normalize_emb_user1, normalize_emb_user2, 0.2) + infoNCE(normalize_emb_item1, normalize_emb_item2, 0.2)
        return self.cl_rate*cl_loss

    def train(self):
        #main task: recommendation
        rec_loss = bpr_loss(self.batch_user_emb,self.batch_pos_item_emb,self.batch_neg_item_emb)
        rec_loss += self.reg * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(self.batch_neg_item_emb))
        #CL task
        self.cl_loss = self.calc_cl_loss()
        loss = rec_loss+self.cl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data,self.batch_size)):
                user_idx, i_idx, j_idx = batch
                _, l, rec_l, cl_l = self.sess.run([train, loss, rec_loss, self.cl_loss],
                                                   feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'total_loss:',l, 'rec_loss:', rec_l,'cl_loss',cl_l)
            self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
            self.training_evaluation(epoch)
        self.U, self.V = self.best_user_emb, self.best_item_emb

    def predict(self, u):
        if self.data.contain_item(u):
            u = self.data.get_user_id(u)
            return self.V.dot(self.U[u])
        else:
            return [0] * self.data.item_num
