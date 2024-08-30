import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface


# paper: Bootstrapping User and Item Representations for One-Class Collaborative Filtering. SIGIR'21


class BUIR(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BUIR, self).__init__(conf, training_set, test_set)
        args = self.config['BUIR']
        self.momentum = float(args['tau'])
        self.n_layers = int(args['n_layer'])
        self.drop_rate = float(args['drop_rate'])
        self.model = BUIR_NB(self.data, self.emb_size, self.momentum, self.n_layers, self.drop_rate, True)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, i_idx, j_idx = batch
                inputs = {'user': user_idx, 'item': i_idx}
                model.train()
                output = model(inputs)
                batch_loss = model.get_loss(output)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                model.update_target(user_idx,i_idx)
                print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            model.eval()
            self.p_u_online, self.u_online, self.p_i_online, self.i_online = self.model.get_embedding()
            self.fast_evaluation(epoch)
        self.p_u_online, self.u_online, self.p_i_online, self.i_online = self.best_p_u, self.best_u, self.best_p_i, self.best_i

    def save(self):
        self.best_p_u, self.best_u, self.best_p_i, self.best_i = self.model.get_embedding()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score_ui = torch.matmul(self.p_u_online[u], self.i_online.transpose(0, 1))
        score_iu = torch.matmul(self.u_online[u], self.p_i_online.transpose(0, 1))
        score = score_ui + score_iu
        return score.cpu().numpy()


class BUIR_NB(nn.Module):
    def __init__(self, data, emb_size, momentum, n_layers, drop_rate, drop_flag=False):
        super(BUIR_NB, self).__init__()
        self.emb_size = emb_size
        self.momentum = momentum
        self.online_encoder = LGCN_Encoder(data, emb_size, n_layers, drop_rate, drop_flag)
        self.target_encoder = LGCN_Encoder(data, emb_size, n_layers, drop_rate, drop_flag)
        self.predictor = nn.Linear(emb_size, emb_size)
        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def update_target(self,u_idx,i_idx):
        # for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
        #     param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
        self.target_encoder.embedding_dict['user_emb'].data[u_idx] = self.target_encoder.embedding_dict['user_emb'].data[u_idx]*self.momentum\
                                                              + self.online_encoder.embedding_dict['user_emb'].data[u_idx]*(1-self.momentum)
        self.target_encoder.embedding_dict['item_emb'].data[i_idx] = self.target_encoder.embedding_dict['item_emb'].data[i_idx]*self.momentum\
                                                              + self.online_encoder.embedding_dict['item_emb'].data[i_idx]*(1-self.momentum)

    def forward(self, inputs):
        u_online, i_online = self.online_encoder(inputs)
        u_target, i_target = self.target_encoder(inputs)
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output
        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)
        return (loss_ui + loss_iu).mean()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, drop_rate, drop_flag=False):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.drop_ratio = drop_rate
        self.drop_flag = drop_flag
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).cuda()
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).cuda()
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        A_hat = self.sparse_dropout(self.sparse_norm_adj, np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        users, items = inputs['user'], inputs['item']
        user_embeddings = user_all_embeddings[users]
        item_embeddings = item_all_embeddings[items]
        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
