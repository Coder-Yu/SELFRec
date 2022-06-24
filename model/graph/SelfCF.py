import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from model.graph.LightGCN import LGCN_Encoder


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
        u_online, i_online = self.online_encoder()
        with torch.no_grad():
            users, items = inputs['user'], inputs['item']
            u_target, i_target = self.u_target_his.clone()[users], self.i_target_his.clone()[items]
            u_target.detach()
            i_target.detach()
            #
            u_target = u_target * self.momentum + u_online[users].data * (1. - self.momentum)
            i_target = i_target * self.momentum + i_online[items].data * (1. - self.momentum)
            #
            self.u_target_his[users, :] = u_online[users].clone()
            self.i_target_his[items, :] = i_online[items].clone()
        return self.predictor(u_online[users]), u_target, self.predictor(i_online[items]), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.forward()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return 1 - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output
        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2
        return loss_ui + loss_iu
