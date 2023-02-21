import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from util.loss_torch import l2_reg_loss, InfoNCE, batch_softmax_loss

# Paper: Self-supervised Learning for Large-scale Item Recommendations. CIKM'21

""" 
Note: This version of code conducts feature dropout on the item embeddings 
because items features are not always available in many academic datasets.
"""


class SSL4Rec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SSL4Rec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SSL4Rec'])
        self.cl_rate = float(args['-alpha'])
        self.tau = float(args['-tau'])
        self.drop_rate = float(args['-drop'])
        self.model = DNN_Encoder(self.data, self.emb_size, self.drop_rate, self.tau)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                query_idx, item_idx, _neg = batch
                model.train()
                query_emb, item_emb = model(query_idx, item_idx)
                rec_loss = batch_softmax_loss(query_emb, item_emb, self.tau)
                cl_loss = self.cl_rate * model.cal_cl_loss(item_idx)
                batch_loss = rec_loss + l2_reg_loss(self.reg, query_emb, item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            model.eval()
            with torch.no_grad():
                self.query_emb, self.item_emb = self.model(list(range(self.data.user_num)),list(range(self.data.item_num)))
            self.fast_evaluation(epoch)
        self.query_emb, self.item_emb = self.best_query_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_query_emb, self.best_item_emb = self.model.forward(list(range(self.data.user_num)),list(range(self.data.item_num)))

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.query_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class DNN_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, temperature):
        super(DNN_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.tau = temperature
        self.user_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(drop_rate)
        initializer = nn.init.xavier_uniform_
        self.initial_user_emb = nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size)))
        self.initial_item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size)))

    def forward(self, q, x):
        q_emb = self.initial_user_emb[q]
        i_emb = self.initial_item_emb[x]

        q_emb = self.user_tower(q_emb)
        i_emb = self.item_tower(i_emb)

        return q_emb, i_emb

    def item_encoding(self, x):
        i_emb = self.initial_item_emb[x]
        i1_emb = self.dropout(i_emb)
        i2_emb = self.dropout(i_emb)

        i1_emb = self.item_tower(i1_emb)
        i2_emb = self.item_tower(i2_emb)

        return i1_emb, i2_emb

    def cal_cl_loss(self, idx):
        item_view1, item_view_2 = self.item_encoding(idx)       
        cl_loss = InfoNCE(item_view1, item_view_2, self.tau)
        return cl_loss
