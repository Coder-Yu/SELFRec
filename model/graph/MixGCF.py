import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss

# paper: MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. KDD'21

#Note: The backbone is LightGCN due to its better performance

class MixGCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MixGCF, self).__init__(conf, training_set, test_set)
        args = self.config['MixGCF']
        self.n_layers = int(args['n_layer'])
        self.n_negs = int(args['n_negs'])
        self.model = MixGCF_Encoder(self.data, self.emb_size, self.n_negs,self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size,self.n_negs)):
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_item_emb, neg_item_emb = model.negative_mixup(user_idx,pos_idx,neg_idx)
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model.get_embeddings()            
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.get_embeddings()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class MixGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_negs,n_layers):
        super(MixGCF_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.layers = n_layers
        self.n_negs = n_negs
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.dropout = nn.Dropout(0.1)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        user_embs = [self.embedding_dict['user_emb']]
        item_embs = [ self.embedding_dict['item_emb']]
        #adj = self._sparse_dropout(self.sparse_norm_adj, 0.5)
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddings = self.dropout(ego_embeddings)
            user_embs.append(ego_embeddings[:self.data.user_num])
            item_embs.append(ego_embeddings[self.data.user_num:])
        user_embs = torch.stack(user_embs, dim=1)
        user_embs = torch.mean(user_embs, dim=1)
        return user_embs, item_embs

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def negative_mixup(self,user,pos_item,neg_item):
        user_emb,item_emb = self.forward()
        u_emb = user_emb[user]
        negs = []
        for k in range(self.layers+1):
            neg_emb = item_emb[k][neg_item]
            pos_emb = item_emb[k][pos_item]
            neg_emb = neg_emb.reshape(-1,self.n_negs,self.emb_size)
            alpha = torch.rand_like(neg_emb).cuda()
            neg_emb = alpha*pos_emb.unsqueeze(dim=1)+(1-alpha)*neg_emb
            scores = ( u_emb.unsqueeze(dim=1)*neg_emb).sum(dim=-1)
            indices = torch.max(scores,dim=1)[1].detach()
            chosen_neg_emb = neg_emb[torch.arange(neg_emb.size(0)), indices]
            negs.append(chosen_neg_emb)
        item_emb = torch.stack(item_emb, dim=1)
        item_emb = torch.mean(item_emb,dim=1)
        negs = torch.stack(negs, dim=1)
        negs = torch.mean(negs, dim=1)
        return u_emb,  item_emb[pos_item], negs

    def get_embeddings(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


