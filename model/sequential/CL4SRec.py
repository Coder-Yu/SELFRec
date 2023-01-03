import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from model.sequential.SASRec import SASRec_Model
from util.loss_torch import l2_reg_loss,InfoNCE,batch_softmax_loss
from data.augmentor import SequenceAugmentor


# Paper: Contrastive Learning for Sequential Recommendation, ICDE'22

class CL4SRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CL4SRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CL4SRec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        head_num = int(args['-n_heads'])
        self.max_len = int(args['-max_len'])
        self.aug_type = int(args['-aug_type'])
        self.aug_rate = float(args['-aug_rate'])
        self.cl_rate = float(args['-cl_rate'])
        self.model = SASRec_Model(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate)
        initializer = nn.init.xavier_uniform_
        self.model.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 2, self.emb_size)))

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, seq_len = batch
                seq_emb = model.forward(seq, pos)
                if self.aug_type == 0: #crop
                    aug_seq1, aug_pos1, aug_len1 = SequenceAugmentor.item_crop(seq, seq_len,self.aug_rate)
                    aug_seq2, aug_pos2, aug_len2 = SequenceAugmentor.item_crop(seq, seq_len, self.aug_rate)
                    aug_emb1 = model.forward(aug_seq1, aug_pos1)
                    aug_emb2 = model.forward(aug_seq2, aug_pos2)
                    cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(aug_len1)]
                    cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(aug_len2)]
                elif self.aug_type == 1: #reorder
                    aug_seq1 = SequenceAugmentor.item_reorder(seq, seq_len, self.aug_rate)
                    aug_seq2 = SequenceAugmentor.item_reorder(seq, seq_len, self.aug_rate)
                    aug_emb1 = model.forward(aug_seq1, pos)
                    aug_emb2 = model.forward(aug_seq2, pos)
                    cl_emb1 = [aug_emb1[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
                    cl_emb2 = [aug_emb2[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
                else : # item mask
                    aug_seq1 = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate,self.data.item_num+1)
                    aug_seq2 = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate,self.data.item_num+1)
                    aug_emb1 = model.forward(aug_seq1, pos)
                    aug_emb2 = model.forward(aug_seq2, pos)
                    cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                    cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                cl_loss = self.cl_rate * InfoNCE(torch.concat(cl_emb1, 0), torch.concat(cl_emb2, 0), 1,False)
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)+cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'rec_loss:', rec_loss.item())
            model.eval()
            self.fast_evaluation(epoch)

    def calculate_loss(self, seq_emb, y, neg,pos):
        y_emb = self.model.item_emb[y]
        neg_emb = self.model.item_emb[neg]
        indices = np.where(pos != 0)
        pos_logits = torch.exp((seq_emb * y_emb).sum(dim=-1))
        neg_logits = torch.exp((seq_emb * neg_emb).sum(dim=-1))
        loss = -torch.log(pos_logits[indices]/(pos_logits[indices]+neg_logits[indices])+10e-6)
        return torch.mean(loss)

    def predict(self, seq):
        current_seq = np.array([self.data.original_seq[seq][-self.max_len:]],dtype=np.int)
        pos = np.array([list(range(1,current_seq.shape[1]+1))],dtype=np.int)
        with torch.no_grad():
            seq_emb = self.model.forward(current_seq,pos)
            last_item_embedding = seq_emb[0,-1,:].unsqueeze(0)
            score = torch.matmul(last_item_embedding, self.model.item_emb[:-1].transpose(0, 1))
        return score.cpu().numpy()


