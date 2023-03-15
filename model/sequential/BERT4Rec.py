import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from model.sequential.SASRec import SASRec_Model
from util.loss_torch import l2_reg_loss
from math import floor
import random


# Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM'19

class BERT4Rec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BERT4Rec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['BERT4Rec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        head_num = int(args['-n_heads'])
        self.aug_rate = float(args['-mask_rate'])
        self.model = SASRec_Model(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate)
        initializer = nn.init.xavier_uniform_
        self.model.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 2, self.emb_size)))
        self.rec_loss = torch.nn.BCEWithLogitsLoss()

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, seq_len = batch
                aug_seq, masked, labels = self.item_mask_for_bert(seq, seq_len, self.aug_rate, self.data.item_num+1)
                seq_emb = model.forward(aug_seq, pos)
                # item mask
                rec_loss = self.calculate_loss(seq_emb,masked,labels)
                batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'rec_loss:', rec_loss.item())
            model.eval()
            self.fast_evaluation(epoch)

    def item_mask_for_bert(self,seq,seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i]*mask_ratio),1))
            masked[i, to_be_masked] = augmented_seq[i, to_be_masked]
            labels += list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq, masked, np.array(labels)

    def calculate_loss(self, seq_emb, masked, labels):
        seq_emb = seq_emb[masked>0].view(-1, self.emb_size)
        logits = torch.mm(seq_emb, self.model.item_emb.t())
        loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())
        return loss

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            seq=np.concatenate([seq,np.zeros((seq.shape[0],1))],axis=1)
            pos=np.concatenate([pos,np.zeros((seq.shape[0],1))],axis=1)
            for i,length in enumerate(seq_len):
                seq[i,length] = self.data.item_num+1
                pos[i,length] = length
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            score = torch.matmul(torch.cat(last_item_embeddings,0), self.model.item_emb.transpose(0, 1))
        return score.cpu().numpy()