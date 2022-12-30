import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor

# Paper: self-supervised graph learning for recommendation. SIGIR'21


class SGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SGL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SGL'])
        self.cl_rate = float(args['-lambda'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch>=5:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1,view2,self.temp)



#========================================================
#tensorflow version

# from base.graph_recommender import GraphRecommender
# from data.augmentor import GraphAugmentor
# import tensorflow as tf
# from base.tf_interface import TFGraphInterface
# from util.loss_tf import bpr_loss, InfoNCE
# from util.conf import OptionConf
# import os
# from util.sampler import next_batch_pairwise
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#
#
# # Paper: self-supervised graph learning for recommendation. SIGIR'21
#
# class SGL(GraphRecommender):
#     def __init__(self, conf, training_set, test_set, **aux_data):
#         super(SGL, self).__init__(conf, training_set, test_set)
#         args = OptionConf(self.config['SGL'])
#         self.ssl_reg = float(args['-lambda'])
#         self.drop_rate = float(args['-droprate'])
#         self.aug_type = int(args['-augtype'])
#         self.ssl_temp = float(args['-temp'])
#         self.n_layers = int(args['-n_layer'])
#
#     def _create_variable(self):
#         self.sub_mat = {}
#         if self.aug_type in [0, 1]:
#             self.sub_mat['adj_values_sub1'] = tf.placeholder(tf.float32)
#             self.sub_mat['adj_indices_sub1'] = tf.placeholder(tf.int64)
#             self.sub_mat['adj_shape_sub1'] = tf.placeholder(tf.int64)
#
#             self.sub_mat['adj_values_sub2'] = tf.placeholder(tf.float32)
#             self.sub_mat['adj_indices_sub2'] = tf.placeholder(tf.int64)
#             self.sub_mat['adj_shape_sub2'] = tf.placeholder(tf.int64)
#         else:
#             for k in range(self.n_layers):
#                 self.sub_mat['adj_values_sub1%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub1%d' % k)
#                 self.sub_mat['adj_indices_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub1%d' % k)
#                 self.sub_mat['adj_shape_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub1%d' % k)
#
#                 self.sub_mat['adj_values_sub2%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub2%d' % k)
#                 self.sub_mat['adj_indices_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub2%d' % k)
#                 self.sub_mat['adj_shape_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub2%d' % k)
#
#     def LightGCN_encoder(self, emb, adj, n_layers):
#         all_embs = [emb]
#         for k in range(n_layers):
#             emb = tf.sparse_tensor_dense_matmul(adj, emb)
#             all_embs.append(emb)
#         all_embs = tf.reduce_mean(all_embs, axis=0)
#         return tf.split(all_embs, [self.data.user_num, self.data.item_num], 0)
#
#     def build(self):
#         super(SGL, self).build()
#         initializer = tf.contrib.layers.xavier_initializer()
#         self.user_embeddings = tf.Variable(initializer([self.data.user_num, self.emb_size]))
#         self.item_embeddings = tf.Variable(initializer([self.data.item_num, self.emb_size]))
#         self.u_idx = tf.placeholder(tf.int32, name="u_idx")
#         self.v_idx = tf.placeholder(tf.int32, name="v_idx")
#         self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
#         self.norm_adj = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj)
#         ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
#         view1_embeddings = ego_embeddings
#         view2_embeddings = ego_embeddings
#         all_view1_embeddings = [view1_embeddings]
#         all_view2_embeddings = [view2_embeddings]
#         all_embeddings = [ego_embeddings]
#         # variable initialization
#         self._create_variable()
#         for k in range(0, self.n_layers):
#             if self.aug_type in [0, 1]:
#                 self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(self.sub_mat['adj_indices_sub1'], self.sub_mat['adj_values_sub1'], self.sub_mat['adj_shape_sub1'])
#                 self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(self.sub_mat['adj_indices_sub2'], self.sub_mat['adj_values_sub2'], self.sub_mat['adj_shape_sub2'])
#             else:
#                 self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(self.sub_mat['adj_indices_sub1%d' % k], self.sub_mat['adj_values_sub1%d' % k], self.sub_mat['adj_shape_sub1%d' % k])
#                 self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(self.sub_mat['adj_indices_sub2%d' % k], self.sub_mat['adj_values_sub2%d' % k], self.sub_mat['adj_shape_sub2%d' % k])
#
#         # augmented view1
#         for k in range(self.n_layers):
#             view1_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_1%d' % k], view1_embeddings)
#             all_view1_embeddings += [view1_embeddings]
#         all_view1_embeddings = tf.stack(all_view1_embeddings, 1)
#         all_view1_embeddings = tf.reduce_mean(all_view1_embeddings, axis=1, keepdims=False)
#         self.view1_user_embeddings, self.view1_item_embeddings = tf.split(all_view1_embeddings, [self.data.user_num, self.data.item_num], 0)
#
#         # augmented view2
#         for k in range(self.n_layers):
#             view2_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_2%d' % k], view2_embeddings)
#             all_view2_embeddings += [view2_embeddings]
#         all_view2_embeddings = tf.stack(all_view2_embeddings, 1)
#         all_view2_embeddings = tf.reduce_mean(all_view2_embeddings, axis=1, keepdims=False)
#         self.view2_user_embeddings, self.view2_item_embeddings = tf.split(all_view2_embeddings, [self.data.user_num, self.data.item_num], 0)
#
#         # recommendation view
#         self.main_user_embeddings, self.main_item_embeddings = self.LightGCN_encoder(ego_embeddings,self.norm_adj,self.n_layers)
#
#         self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
#         self.batch_neg_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
#         self.batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
#         self.batch_pos_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)
#         tf_config = tf.ConfigProto()
#         tf_config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=tf_config)
#
#     def calc_ssl_loss(self):
#         user_emb1 = tf.nn.embedding_lookup(self.view1_user_embeddings, tf.unique(self.u_idx)[0])
#         user_emb2 = tf.nn.embedding_lookup(self.view2_user_embeddings, tf.unique(self.u_idx)[0])
#         item_emb1 = tf.nn.embedding_lookup(self.view1_item_embeddings, tf.unique(self.v_idx)[0])
#         item_emb2 = tf.nn.embedding_lookup(self.view2_item_embeddings, tf.unique(self.v_idx)[0])
#         emb_merge1 = tf.concat([user_emb1, item_emb1], axis=0)
#         emb_merge2 = tf.concat([user_emb2, item_emb2], axis=0)
#         normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
#         normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
#         ssl_loss = self.ssl_reg * InfoNCE(normalize_emb_merge1, normalize_emb_merge2, 0.2)
#         return ssl_loss
#
#     def train(self):
#         # main task: recommendation
#         rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
#         rec_loss += self.reg * (
#                     tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
#                 self.batch_neg_item_emb))
#         # SSL task: contrastive learning
#         ssl_loss = self.calc_ssl_loss()
#         total_loss = rec_loss + ssl_loss
#
#         opt = tf.train.AdamOptimizer(self.lRate)
#         train = opt.minimize(total_loss)
#
#         init = tf.global_variables_initializer()
#         self.sess.run(init)
#         for epoch in range(self.maxEpoch):
#             sub_mat = {}
#             if self.aug_type == 0:
#                 dropped_mat1 = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
#                 adj_mat1 = self.data.convert_to_laplacian_mat(dropped_mat1)
#                 sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat[
#                     'adj_shape_sub1'] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat1)
#
#                 dropped_mat2 = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
#                 adj_mat2 = self.data.convert_to_laplacian_mat(dropped_mat2)
#                 sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat[
#                     'adj_shape_sub2'] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat2)
#
#             elif self.aug_type == 1:
#                 dropped_mat1 = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
#                 adj_mat1 = self.data.convert_to_laplacian_mat(dropped_mat1)
#                 sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat[
#                     'adj_shape_sub1'] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat1)
#
#                 dropped_mat2 = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
#                 adj_mat2 = self.data.convert_to_laplacian_mat(dropped_mat2)
#                 sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat[
#                     'adj_shape_sub2'] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat2)
#             else:
#                 for k in range(self.n_layers):
#                     dropped_mat1 = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
#                     adj_mat1 = self.data.convert_to_laplacian_mat(dropped_mat1)
#                     sub_mat['adj_indices_sub1%d' % k], sub_mat['adj_values_sub1%d' % k], sub_mat[
#                         'adj_shape_sub1%d' % k] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat1)
#                     dropped_mat2 = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
#                     adj_mat2 = self.data.convert_to_laplacian_mat(dropped_mat2)
#                     sub_mat['adj_indices_sub2%d' % k], sub_mat['adj_values_sub2%d' % k], sub_mat[
#                         'adj_shape_sub2%d' % k] = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(adj_mat2)
#
#             for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
#                 user_idx, i_idx, j_idx = batch
#                 feed_dict = {self.u_idx: user_idx,
#                              self.v_idx: i_idx,
#                              self.neg_idx: j_idx, }
#                 if self.aug_type in [0, 1]:
#                     feed_dict.update({
#                         self.sub_mat['adj_values_sub1']: sub_mat['adj_values_sub1'],
#                         self.sub_mat['adj_indices_sub1']: sub_mat['adj_indices_sub1'],
#                         self.sub_mat['adj_shape_sub1']: sub_mat['adj_shape_sub1'],
#                         self.sub_mat['adj_values_sub2']: sub_mat['adj_values_sub2'],
#                         self.sub_mat['adj_indices_sub2']: sub_mat['adj_indices_sub2'],
#                         self.sub_mat['adj_shape_sub2']: sub_mat['adj_shape_sub2']
#                     })
#                 else:
#                     for k in range(self.n_layers):
#                         feed_dict.update({
#                             self.sub_mat['adj_values_sub1%d' % k]: sub_mat['adj_values_sub1%d' % k],
#                             self.sub_mat['adj_indices_sub1%d' % k]: sub_mat['adj_indices_sub1%d' % k],
#                             self.sub_mat['adj_shape_sub1%d' % k]: sub_mat['adj_shape_sub1%d' % k],
#                             self.sub_mat['adj_values_sub2%d' % k]: sub_mat['adj_values_sub2%d' % k],
#                             self.sub_mat['adj_indices_sub2%d' % k]: sub_mat['adj_indices_sub2%d' % k],
#                             self.sub_mat['adj_shape_sub2%d' % k]: sub_mat['adj_shape_sub2%d' % k]
#                         })
#
#                 _, l, rec_l, ssl_l = self.sess.run([train, total_loss, rec_loss, ssl_loss], feed_dict=feed_dict)
#                 if n % 100 == 0:
#                     print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_l, 'ssl_loss', ssl_l)
#             self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
#             self.fast_evaluation(epoch)
#         self.U, self.V = self.best_user_emb, self.best_item_emb
#
#     def save(self):
#         self.best_user_emb, self.best_item_emb = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
#
#     def predict(self, u):
#         u = self.data.get_user_id(u)
#         return self.V.dot(self.U[u])
