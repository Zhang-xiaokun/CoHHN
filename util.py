import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) #count the unique items in a session，delete the repeat items, ranking by item_id
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    # indptr:sum of the session length; indices:item_id - 1
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 10000 * 6558 #sessions * #items H
    return matrix

def data_easy_masks(data_l, n_row, n_col):
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]

    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 10000 * 6558 #sessions * #items H
    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, n_node=None, n_price=None, n_category=None):
        self.raw = np.asarray(data[0]) # sessions, item_seq

        self.price_raw = np.asarray(data[1]) # price_seq

        H_T = data_easy_masks(data[2], len(data[0]), n_node)  # 10000 * 6558 #sessions * #items
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        H_pv = data_easy_masks(data[4], n_price, n_node)
        BH_pv = H_pv

        BH_vp = H_pv.T


        H_pc = data_easy_masks(data[5], n_price, n_category)
        BH_pc = H_pc

        BH_cp = H_pc.T

        H_cv = data_easy_masks(data[6], n_category, n_node)
        BH_cv = H_cv
        
        BH_vc = H_cv.T


        self.adjacency = DHBH_T.tocoo()

        self.adjacency_pv = BH_pv.tocoo()
        self.adjacency_vp = BH_vp.tocoo()
        self.adjacency_pc = BH_pc.tocoo()
        self.adjacency_cp = BH_cp.tocoo()
        self.adjacency_cv = BH_cv.tocoo()
        self.adjacency_vc = BH_vc.tocoo()

        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.targets = np.asarray(data[7])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # random session item_seq & price_seq
            self.raw = self.raw[shuffled_arg]
            self.price_raw = self.price_raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node, price_seqs = [], [], []
        inp = self.raw[index]
        inp_price = self.price_raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session, price in zip(inp,inp_price):
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, price_seqs


