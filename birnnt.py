import torch
import torch.nn as nn
from birnn import BiRNN
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
from Util import IndexLinear

# use_cuda = False
import random
import gc

class BiRNNT(BiRNN):
    def __init__(self, v_size, t_size, u_size, emb_dim_v, emb_dim_t, emb_dim_u, emb_dim_d, hidden_dim, nb_cnt, candidate,vid_coor_nor):
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
        self.decoder_dim = self.hidden_dim * 2 + self.emb_dim_t + self.emb_dim_u + self.nb_cnt
        self.decoder = nn.Linear(self.decoder_dim, self.v_size)
        self.decoder2 = nn.Linear(self.nb_cnt,self.v_size)
        self.candidate = candidate
        self.vid_coor_nor = vid_coor_nor
        self.linear_d1 = nn.Linear(v_size,1)
        self.embedder_d2 = nn.Embedding(v_size,self.emb_dim_d,padding_idx = 0)
        self.att_merger = nn.Linear(2,1)
        self.decoder_hl = IndexLinear(self.hidden_dim, v_size)
        self.decoder_hs = IndexLinear(self.hidden_dim, v_size)
        self.decoder_t = IndexLinear(self.emb_dim_t, v_size)
        self.decoder_u = IndexLinear(self.emb_dim_u, v_size)
        self.merger_weight = nn.Parameter(torch.ones(1, 5) / 5.0)


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

    def get_d_all(self,uids, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate,feature_al): #jiayi
        n = 0
        atten_scores = Variable(torch.zeros((1, max(len_long))))
        scores_d = np.zeros((sum(len_long)-len(len_long),self.nb_cnt)) #just a try
        m = max(len_long)
        delete_idx = []
        vid_candidates_all = []
        for idx,length in enumerate(len_long): # every user records_u
            #print(idx) # the problems lies in maxs
            # if length >200:
            #      n += (length - 1)
            #      delete_idx.append(idx)
            #      continue
            if length == 406:
                print(1)
            feature_temp = torch.mm(feature_al[n + 1: n + length], feature_al[n:n + length].t())  # all the feature part for every userdist
            print(idx)
            #dist_temp = Variable(torch.zeros((length-1,length)))
            for i in range(length-1): # the vids of every user
                for j in range(length): # single vid
                    '''
                    dist_temp[i,j] = self.get_distance(vids_long[idx][i],vids_long[idx][j])
                    atten_scores[n+i,j] = self.att_merger(torch.cat((feature_temp[i,j:j+1],dist_temp[i,j:j+1])))
                    '''
                    dist_temp = self.get_distance(vids_long[idx][i], vids_long[idx][j]) # for this I directly delete 3555
                    try:
                        atten_scores[0, j] = self.att_merger(torch.cat((feature_temp[i, j:j + 1], torch.FloatTensor([dist_temp]))))
                    except:
                        print('bug atten score')
                atten_scores[0,i+1] = float('-inf')
                atten_scores[0,0:j+1] = F.softmax(atten_scores[0,0:j+1]) #atten_scores inf solved
                #in LP3 everything is based on vid candidate size but LP4 is not!
                #score_d for the candidate of every vid
                vid_candidates = self.get_vids_candidate(int(vids_long[idx][i]),int(vids_long[idx][i+1]))
                vid_candidates_all.append(vid_candidates)
                #dist = np.zeros((len(vid_candidates), length))
                dist = np.zeros((1, length))
                temp_atten = atten_scores[0, 0:length]
                for j, vid_candidate in enumerate(vid_candidates):
                    for k in range(length):
                        #dist[j][k] = float(np.exp(-self.get_distance(vid_candidates[j], vids_long[idx][k])))
                        dist[0][k] = float(np.exp(-self.get_distance(vid_candidates[j], vids_long[idx][k])))
                 #idx 49 i = 3
                # try:
                #     scores_d[n + i] = np.dot(dist , temp_atten.data.numpy().T)
                # except:
                #     temp_atten = atten_scores[n+i,0:length][0]
                #     scores_d[n + i] = np.dot(dist, temp_atten.data.numpy().T)
                    try:
                        scores_d[n + i,j] = np.dot(dist , temp_atten.data.numpy().T)
                    except:
                        temp_atten = atten_scores[0,0:length][0]
                        scores_d[n + i,j] = np.dot(dist, temp_atten.data.numpy().T)


            n+=(length-1)

        #gc.set_debug(gc.DEBUG_LEAK)
        #scores_d = np.delete(scores_d,delete_idx,0)
        # del(atten_scores)
        # del(dist)
        # gc.collect()
        # print(1)
        return Variable(torch.from_numpy(scores_d)).float(),delete_idx,vid_candidates_all

    def get_vids_candidate(self,vid_current,vid_next):
        vids, probs_pop, probs_dist = self.candidate[vid_current-1]
        if random.random() < 0.5:
            id_cnt = np.random.multinomial(self.nb_cnt-1, probs_pop) # waiting to add self.nb_cnt 1111
        else:
            id_cnt = np.random.multinomial(self.nb_cnt-1, probs_dist)
        vid_candidates = [vid_next]
        for id, cnt in enumerate(id_cnt):
            for _ in range(cnt):
                vid_candidates.append(vids[id])
        return vid_candidates

    def get_d(self,vid_r,vid_cand):
        coor_diff = self.vid_coor_nor[vid_cand] - self.vid_coor_nor[vid_r]
        return float(np.exp(-np.sqrt(np.sum(coor_diff ** 2))))

    def forward(self, uids, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate):
        mask_long_valid = mask_long.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        #mask_optim_valid = (mask_optim if len(mask_evaluate) == 0 else mask_evaluate).index_select(1, Variable(torch.LongTensor(xrange(torch.max(len_long).data[0])))).masked_select(mask_long_valid)
        embeddings_t = self.get_embedding_t(tids_next, len_long, mask_long_valid)
        embeddings_u = self.get_embeddeing_u(uids, len_long, mask_long_valid)
        hiddens_long = self.get_hiddens_long(vids_long, len_long, mask_long_valid)
        hiddens_short = self.get_hiddens_short(vids_short_al, len_short_al,short_cnt)
        feature_al = torch.cat((F.relu(hiddens_long), F.relu(hiddens_short), F.relu(embeddings_t)), 1)
        # get delete mask_optim_valid
        len_long_d = len_long - 1
        mask_long_valid_d = mask_long.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long_d).data[0]))))
        lm = len(mask_long_valid_d[0])
        for idx, l in enumerate(len_long_d):
            if l < lm:
                    mask_long_valid_d[idx][l] = 0
        mask_optim_valid_d = (mask_optim if len(mask_evaluate) == 0 else mask_evaluate).index_select(1, Variable(torch.LongTensor(xrange(torch.max(len_long_d).data[0])))).masked_select(mask_long_valid_d)
        print("begin d")
        d,delete_long,vid_candidates_all = self.get_d_all(uids, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long,mask_optim, mask_evaluate, feature_al)
        #gc.set_debug(gc.DEBUG_LEAK)
        print("end d")
        #delete
        idx_delete = np.cumsum(len_long)-1 #last idx to delete
        '''
        idx = torch.LongTensor(np.delete(range(len(hiddens_long)),idx_delete)) #idx to keep (in case
        embeddings_t_delete = torch.index_select(embeddings_t,0,idx)
        embeddings_u_delete = torch.index_select(embeddings_u,0, idx)
        hiddens_long_delete = torch.index_select(hiddens_long,0,idx)
        hiddens_short_delete = torch.index_select(hiddens_long,0, idx)
        #other with candidate
        scores_t = self.decoder_t(embeddings_t_delete, Variable(torch.LongTensor(vid_candidates_all)))
        scores_u = self.decoder_u(embeddings_u_delete, Variable(torch.LongTensor(vid_candidates_all)))
        scores_hl = self.decoder_hl(hiddens_long_delete,Variable(torch.LongTensor(vid_candidates_all)))
        scores_hs = self.decoder_hs(hiddens_short_delete,Variable(torch.LongTensor(vid_candidates_all)))
        '''
        #gc solution
        idx = torch.LongTensor(np.delete(range(len(hiddens_long)), idx_delete))  # idx to keep (in case
        embeddings_t = torch.index_select(embeddings_t, 0, idx)
        embeddings_u = torch.index_select(embeddings_u, 0, idx)
        hiddens_long = torch.index_select(hiddens_long, 0, idx)
        hiddens_short = torch.index_select(hiddens_short, 0, idx)
        # other with candidate
        scores_t = self.decoder_t(embeddings_t, Variable(torch.LongTensor(vid_candidates_all)))
        scores_u = self.decoder_u(embeddings_u, Variable(torch.LongTensor(vid_candidates_all)))
        scores_hl = self.decoder_hl(hiddens_long, Variable(torch.LongTensor(vid_candidates_all)))
        scores_hs = self.decoder_hs(hiddens_short, Variable(torch.LongTensor(vid_candidates_all)))

        #combine
        #hiddens_comb = torch.cat((hiddens_long, hiddens_short, embeddings_t,embeddings_u,d), 1)
        new_comb = torch.cat((scores_t,scores_u,scores_hl,scores_hs,d.view(len(idx),1,self.nb_cnt)),1)
        #predicted_scores = F.sigmoid(F.linear(new_comb, F.relu(self.merger_weight), bias=None).t())
        predicted_scores = Variable(torch.zeros(len(idx), self.nb_cnt))
        for i in range(len(idx)):
            predicted_scores[i] = F.sigmoid(F.linear(new_comb[i].t(), self.merger_weight, bias=None).t())
            print(F.linear)

        #hiddens_comb2 = torch.cat((hiddens_long_delete, hiddens_short_delete, embeddings_t_delete,embeddings_u_delete,d), 1) #jiayi
        #bug mask jiayi 1111
        #mask_optim_expanded2 = mask_optim_valid_d.view(-1, 1).expand_as(hiddens_comb2)
        #hiddens_comb_masked2 = hiddens_comb2.masked_select(mask_optim_expanded2).view(-1, self.decoder_dim) #decoder_dim connect to every maxlen_long? jiayi
        #That's why they set vid_candidates, to unify the length
        #decoded = self.decoder(hiddens_comb_masked2)
        #jiayi waiting111 linear (decoded= score_utdh_comb, need a linear to make it score and combine)


        mask_optim_expanded = mask_optim_valid_d.view(-1, 1).expand_as(predicted_scores)
        hiddens_comb_masked = predicted_scores.masked_select(mask_optim_expanded).view(-1, self.nb_cnt) #decoder_dim connect to every maxlen_long? jiayi
        vid_candidates_masked = Variable(torch.LongTensor(vid_candidates_all)).masked_select(mask_optim_expanded).view(
            -1, self.nb_cnt)
        #decoded2 = self.decoder2(hiddens_comb_masked)

        #a,ab = decoded.sort(1,descending = True)
        #c,cd = decoded2.sort(1,descending = True)
        #return predicted_scores,decoded,decoded2  # return scores_merge instead
        #return decoded2 #jiayi
        # del(feature_al)
        # del(scores_t)
        # del(scores_u)
        # del(scores_hs)
        # del(scores_hl)
        # del(new_comb)
        # del(predicted_scores)
        # gc.collect()
        return hiddens_comb_masked,vid_candidates_masked




    def get_scores_d_all(self,vids_long, len_long, mask_long_valid):
        '''
        distance_vids_score = Variable(torch.zeros(self.v_size,1))
        for idx,(len_long_vid,vid) in enumerate(zip(len_long,vids_long)):
            distance_vid_score = self.get_scores_d(vid,len_long_vid)
            distance_vids_score = torch.cat((distance_vids_score,distance_vid_score),1)
        #ds = Variable(torch.FloatTensor(distance_vids_score))
        distance_vids_score = np.delete(distance_vids_score.data.numpy(),0,1)
        #distance_vids_score = Variable(torch.LongTensor(distance_vids_score))
        distance_vids_score = Variable(torch.FloatTensor(distance_vids_score))
        ds = torch.t(distance_vids_score)
        ds_strip = ds.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_d = self.embedder_d2(ds_strip).view(-1, self.emb_dim_d)
        embedding_d_valid = embedding_d.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_d)).view(-1, self.emb_dim_d)
        '''
        distance_vids_score = Variable(torch.zeros(self.v_size,1))
        for idx,(len_long_vid,vid) in enumerate(zip(len_long,vids_long)):
            distance_vid_score = self.get_scores_d(vid,len_long_vid)
            distance_vids_score = torch.cat((distance_vids_score,distance_vid_score),1)
        #ds = Variable(torch.FloatTensor(distance_vids_score))
        distance_vids_score = np.delete(distance_vids_score.data.numpy(),0,1) #49*5670 np *10000

        '''
        distance_vids_score1 = Variable(torch.LongTensor(distance_vids_score))  # 5671*50 long
        ds1 = torch.t(distance_vids_score1)
        ds_strip1 = ds1.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_d1 = self.embedder_d2(ds_strip1).view(-1, self.emb_dim_d)
        embedding_d_valid1 = embedding_d1.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_d1)).view(-1,self.emb_dim_d)
        '''
        a = np.full((len(distance_vids_score), len(distance_vids_score[0])), 100)
        m = distance_vids_score.min()
        if m < 0:
            b = np.full((len(distance_vids_score), len(distance_vids_score[0])), -m)
            distance_vids_score = (distance_vids_score + b) * a
        else:
            distance_vids_score = distance_vids_score * a
        distance_vids_score = Variable(torch.LongTensor(distance_vids_score)) #5671*50 long

        ds = torch.t(distance_vids_score)
        ds_strip = ds.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_d = self.embedder_d2(ds_strip).view(-1, self.emb_dim_d)
        embedding_d_valid = embedding_d.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_d)).view(-1, self.emb_dim_d)
        return embedding_d_valid

    def get_scores_d(self,vid,len_long_vid):
        vid = vid.data.numpy()
        vid_d = np.zeros((self.v_size,self.v_size))
        for i in range(len_long_vid):
            for j in range(i,len_long_vid):
                vid_d[i][j] = self.get_distance(vid[i],vid[j])
                vid_d[j][i] = vid_d[i][j]
        vid_d = Variable(torch.t(torch.FloatTensor(vid_d)))
        distance_vid_score = self.linear_d1(vid_d)
        return distance_vid_score

    def get_distance(self,v1,v2):
        '''
        if v1<v2:
            d = self.distance[v1-1][v2-v1-1]
        elif v1>v2:
            d = self.distance[v2-1][v1-v2-1]
        else:
            d = 0
        return d
        '''
        try:
            return np.sqrt(np.sum((self.vid_coor_nor[v1-1] - self.vid_coor_nor[v2-1]) ** 2))
        except:
            return 0


