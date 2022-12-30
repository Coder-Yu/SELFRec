import torch
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss
from model.graph.MF import Matrix_Factorization
from model.graph.LightGCN import LGCN_Encoder

class DirectAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DirectAU, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DirectAU'])
        self.gamma = float(args['-gamma'])
        self.n_layers= int(args['-n_layers'])
        self.model = LGCN_Encoder(self.data, self.emb_size,self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                batch_loss = self.calculate_loss(user_emb, pos_item_emb)+ l2_reg_loss(self.reg, user_emb,pos_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self,user_emb,item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = self.gamma * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + uniform

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()