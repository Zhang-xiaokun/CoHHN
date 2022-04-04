import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, n_category):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category

        self.mat_pc = nn.Parameter(torch.Tensor(self.n_price, self.emb_size))
        self.mat_cp = nn.Parameter(torch.Tensor(self.n_category, 1))

        self.mat_pc = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_cv = nn.Parameter(torch.Tensor(self.n_category, 1))


        self.a_o_g_i = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gi1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_p = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gp1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_c = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gc1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc2 = nn.Linear(self.emb_size, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency,adjacency_pv,adjacency_vp, adjacency_pc, adjacency_cp, adjacency_cv, adjacency_vc, embedding, pri_emb, cate_emb):
        for i in range(self.layers):
            item_embeddings = self.inter_gate(self.a_o_g_i, self.b_o_gi1, self.b_o_gi2, embedding, self.get_embedding(adjacency_vp, pri_emb) ,
                self.get_embedding(adjacency_vc, cate_emb))  + self.get_embedding(adjacency, embedding)

            price_embeddings = self.inter_gate(self.a_o_g_p, self.b_o_gp1, self.b_o_gp2, pri_emb,
                                                         self.intra_gate(adjacency_pv, self.mat_pv, embedding),
                                                         self.intra_gate(adjacency_pc, self.mat_pc, cate_emb))

            category_embeddings =  self.inter_gate(self.a_o_g_c, self.b_o_gc1, self.b_o_gc2, cate_emb,
                                                             self.intra_gate(adjacency_cp, self.mat_cp, pri_emb),
                                                             self.intra_gate(adjacency_cv, self.mat_cv, embedding))
            embedding = item_embeddings
            pri_emb = price_embeddings
            cate_emb = category_embeddings

        return item_embeddings, price_embeddings

    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        embs = embedding
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings
    def intra_gate(self, adjacency, mat_v, embedding2):
        # attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        matrix = adjacency.to_dense().cuda()
        mat_v = mat_v.expand(mat_v.shape[0], self.emb_size)
        alpha = torch.mm(mat_v, torch.transpose(embedding2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        alpha = alpha * matrix
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row
        type_embs = torch.mm(alpha, embedding2)
        item_embeddings = type_embs
        return self.dropout70(item_embeddings)
    def inter_gate(self, a_o_g, b_o_g1, b_o_g2, emb_mat1, emb_mat2, emb_mat3):
        all_emb1 = torch.cat([emb_mat1, emb_mat2, emb_mat3], 1)
        gate1 = torch.sigmoid(a_o_g(all_emb1) + b_o_g1(emb_mat2) + b_o_g2(emb_mat3))
        h_embedings = emb_mat1 + gate1 * emb_mat2 + (1 - gate1) * emb_mat3
        return self.dropout50(h_embedings)


class CoHHN(Module):
    def __init__(self, adjacency, adjacency_pv, adjacency_vp,adjacency_pc,adjacency_cp,adjacency_cv,adjacency_vc, n_node, n_price, n_category, lr, layers, l2, beta, dataset, num_heads=4, emb_size=100, batch_size=100):
        super(CoHHN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta

        self.adjacency = adjacency
        self.adjacency_pv = adjacency_pv
        self.adjacency_vp = adjacency_vp
        self.adjacency_pc = adjacency_pc
        self.adjacency_cp = adjacency_cp
        self.adjacency_cv = adjacency_cv
        self.adjacency_vc = adjacency_vc


        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.price_embedding = nn.Embedding(self.n_price, self.emb_size)
        self.category_embedding = nn.Embedding(self.n_category, self.emb_size)


        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset,self.emb_size, self.n_node, self.n_price, self.n_category)

        self.w_1 = nn.Linear(self.emb_size*2, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # self_attention
        if emb_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
        # parameters setting
        self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16  the dimension of attention head
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value
        self.query = nn.Linear(self.emb_size , self.emb_size )  # 128, 128
        self.key = nn.Linear(self.emb_size , self.emb_size )
        self.value = nn.Linear(self.emb_size , self.emb_size )

        # co-guided networks
        self.w_p_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.u_i_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # gate5 & gate6
        self.w_pi_1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_pi_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_i = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)



        self.mlp_m_p_1 =  nn.Linear(self.emb_size*2, self.emb_size, bias=True)
        self.mlp_m_i_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)

        self.mlp_m_p_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.mlp_m_i_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, price_embedding, session_item, price_seqs, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)  # for different GPU
        mask = mask.float().unsqueeze(-1)

        price_embedding = torch.cat([zeros, price_embedding], 0)
        get_pri = lambda i: price_embedding[price_seqs[i]]
        seq_pri = torch.cuda.FloatTensor(self.batch_size, list(price_seqs.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size) # for different GPU
        for i in torch.arange(price_seqs.shape[0]):
            seq_pri[i] = get_pri(i)

        # self-attention to get price preference
        attention_mask = mask.permute(0,2,1).unsqueeze(1)  # [bs, 1, 1, seqlen]
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(seq_pri)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_pri)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_pri)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores + attention_mask
        # add mask，set padding to -10000
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # [bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)
        item_pos = torch.tensor(range(1, seq_pri.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(price_seqs)

        item_pos = item_pos * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        price_pre = torch.sum(last_interest, 1)

        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)

        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        beta = beta * mask
        interest_pre = torch.sum(beta * seq_h, 1)

        # Co-guided Learning
        m_c = torch.tanh(self.w_pi_1(price_pre * interest_pre))
        m_j = torch.tanh(self.w_pi_2(price_pre + interest_pre))

        r_i = torch.sigmoid(self.w_c_z(m_c) + self.u_j_z(m_j))
        r_p = torch.sigmoid(self.w_c_r(m_c) + self.u_j_r(m_j))

        m_p = torch.tanh(self.w_p(price_pre * r_p) + self.u_p((1 - r_p) * interest_pre))
        m_i = torch.tanh(self.w_i(interest_pre * r_i) + self.u_i((1 - r_i) * price_pre))

        # enriching the semantics of price and interest preferences
        p_pre = (price_pre + m_i )* m_p
        i_pre = (interest_pre + m_p) * m_i

        return i_pre, p_pre
    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def forward(self, session_item, price_seqs, session_len, reversed_sess_item, mask):
        # session_item all sessions in a batch [[23,34,0,0],[1,3,4,0]]
        item_embeddings_hg, price_embeddings_hg = self.HyperGraph(self.adjacency, self.adjacency_pv, self.adjacency_vp, self.adjacency_pc, self.adjacency_cp, self.adjacency_cv, self.adjacency_vc, self.embedding.weight, self.price_embedding.weight, self.category_embedding.weight) # updating the item embeddings
        sess_emb_hgnn, sess_pri_hgnn = self.generate_sess_emb(item_embeddings_hg, price_embeddings_hg, session_item, price_seqs, session_len, reversed_sess_item, mask) # session embeddings in a batch
        # get item-price table return price of items
        v_table = self.adjacency_vp.row
        temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
        vp_idx = self.adjacency_vp.col[idx]
        item_pri_l = price_embeddings_hg[vp_idx]

        return item_embeddings_hg, price_embeddings_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l


def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs = data.get_slice(i) # obtaining instances from a batch
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l = model(session_item, price_seqs, session_len, reversed_sess_item, mask)
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(item_pri_l, 1, 0))
    scores = scores_interest + scores_price
    return tar, scores


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in slices:
        model.zero_grad()
        targets, scores = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss


