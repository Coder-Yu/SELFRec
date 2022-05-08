import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface


# SELFCF: A Simple Framework for Self-supervised Collaborative Filtering

# Note: SelfCF has three variants and We implement SelfCF-he because it is reported as the best in most cases. The backbone is LightGCN.

class SelfCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SelfCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SelfCF'])
        self.momentum = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.model = SelfCF_HE(self.data, self.emb_size, self.momentum, self.n_layers)

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


class SelfCF_HE(nn.Module):
    def __init__(self, data, emb_size, momentum, n_layers):
        super(SelfCF_HE, self).__init__()
        self.user_count = data.user_num
        self.item_count = data.item_num
        self.latent_size = emb_size
        self.momentum = momentum
        self.online_encoder = LGCN_Encoder(data, emb_size, n_layers)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.u_target_his = torch.randn((self.user_count, self.latent_size), requires_grad=False).cuda()
        self.i_target_his = torch.randn((self.item_count, self.latent_size), requires_grad=False).cuda()

    def forward(self, inputs):
        u_online, i_online = self.online_encoder(inputs)
        with torch.no_grad():
            users, items = inputs['user'], inputs['item']
            u_target, i_target = self.u_target_his.clone()[users, :], self.i_target_his.clone()[items, :]
            u_target.detach()
            i_target.detach()
            #
            u_target = u_target * self.momentum + u_online.data * (1. - self.momentum)
            i_target = i_target * self.momentum + i_online.data * (1. - self.momentum)
            #
            self.u_target_his[users, :] = u_online.clone()
            self.i_target_his[items, :] = i_online.clone()
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return 1 - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output
        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2
        return loss_ui + loss_iu


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self, inputs):
        A_hat =  self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num, :]
        item_all_embeddings = all_embeddings[self.data.user_num:, :]
        users, items = inputs['user'], inputs['item']
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]
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
        user_all_embeddings = all_embeddings[:self.data.user_num, :]
        item_all_embeddings = all_embeddings[self.data.user_num:, :]
        return user_all_embeddings, item_all_embeddings
