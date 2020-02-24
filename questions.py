import torch
import torch.nn as nn
from birnn import BiRNN
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
# use_cuda = False
import random

class BiRNNT(BiRNN):
    def __init__(self, v_size, t_size, u_size, emb_dim_v, emb_dim_t, emb_dim_u, emb_dim_d, hidden_dim, nb_cnt, distance,candidate):
        super(BiRNNT, self).__init__(v_size, emb_dim_v, hidden_dim)
        self.t_size = t_size
        self.v_size = v_size
        self.emb_dim_t = emb_dim_t
        self.emb_dim_u = emb_dim_u
        self.emb_dim_d = emb_dim_d
        self.nb_cnt = nb_cnt
        self.embedder_t = nn.Embedding(t_size, self.emb_dim_t, padding_idx=0)
        self.embedder_u = nn.Embedding(u_size,self.emb_dim_u,padding_idx=0)
        #self.embedder_lat = nn.Embedding(l_size,self.emb_dim_l,padding_idx=0)
        #self.embedder_lon = nn.Embedding(l_size,self.embd_dim_l,padding_idx=0)
        self.decoder_dim = self.hidden_dim * 2 + self.emb_dim_t + self.emb_dim_u + self.emb_dim_d
        self.decoder = nn.Linear(self.decoder_dim, self.v_size)
        self.distance = distance
        self.candidate = candidate
        self.linear_d1 = nn.Linear(v_size,1)
        self.embedder_d2 = nn.Embedding(v_size,self.emb_dim_d,padding_idx = 0)
        self.att_merger = nn.Linear(2,1)

    def get_embeddeing_u(self, uids, len_long, mask_long_valid):
        uids_strip = uids.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_u = self.embedder_u(uids_strip).view(-1, self.emb_dim_u)
        embedding_u_valid = embedding_u.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_u)).view(-1, self.emb_dim_u)
        return embedding_u_valid

    def get_embedding_t(self, tids_next, len_long, mask_long_valid):
        tids_next_strip = tids_next.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_t = self.embedder_t(tids_next_strip).view(-1, self.emb_dim_t)
        embedding_t_valid = embedding_t.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_t)).view(-1, self.emb_dim_t)
        return embedding_t_valid

    def get_d_all(self,uids, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate,feature_al,atten_scores): #jiayi
        n = 0
        scores_d = Variable(torch.zeros((sum(len_long)-len(len_long),self.nb_cnt))) #just a try
        for idx,length in enumerate(len_long): # every user records_u
            for i in range(length-1): # the vids of every user
                vid_candidates = self.get_vids_candidate(int(vids_long[idx][i]),int(vids_long[idx][i+1]))
                for j, vid_candidate in enumerate(vid_candidates):
                    score_sum = Variable(torch.zeros([1]))
                    for k in range(length):
                        if j == k+1:
                            continue
                        score = float(np.exp(-self.get_distance(vid_candidates[j], vids_long[idx][k])))
                        score_sum = score_sum + atten_scores[n + i, k] * score
                    scores_d[n + i,j] = score
            n+=(length-1)
        return scores_d




