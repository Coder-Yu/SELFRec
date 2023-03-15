import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.structure import PointWiseFeedForward
from util.loss_torch import l2_reg_loss


# Paper: Self-Attentive Sequential Recommendation
# Code Referred: https://github.com/pmixer/SASRec.pytorch/


class SASRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SASRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SASRec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        head_num = int(args['-n_heads'])
        self.model = SASRec_Model(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, _ = batch
                seq_emb = model.forward(seq, pos)
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', batch_loss.item())
            model.eval()
            self.fast_evaluation(epoch)

    def calculate_loss(self, seq_emb, y, neg,pos):
        y_emb = self.model.item_emb[y]
        neg_emb = self.model.item_emb[neg]
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
        indices = np.where(pos != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            score = torch.matmul(torch.cat(last_item_embeddings,0), self.model.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SASRec_Model(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate):
        super(SASRec_Model, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num+1, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len+1, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer =  torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seq, pos):
        seq_emb = self.item_emb[seq]
        seq_emb *= self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb += pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        seq_emb *= ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            #attention_input = seq_emb
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb *=  ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb
