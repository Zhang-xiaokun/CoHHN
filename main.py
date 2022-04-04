'''

creat by kun at Oct 2021
Reference: https://github.com/xiaxin1998/DHCN
'''



import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon', help='dataset name: amazon/digineticaBuy/cosmetics/')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=2, help='the number of layer used, 2 for amazon, 3 for others')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(0)

def main():
    # list[0]:session list[1]:label
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

    if  opt.dataset == 'cosmetics':
        n_node = 23194
        n_price = 11
        n_category = 301
    elif opt.dataset == 'digineticaBuy':
        n_node = 24889
        n_price = 100
        n_category = 721
    elif opt.dataset == 'amazon':
        n_node = 9114
        n_price = 51
        n_category = 613
    else:
        print("unkonwn dataset")
    # data_formate: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv
    train_data = Data(train_data, shuffle=True, n_node=n_node, n_price=n_price, n_category=n_category)
    test_data = Data(test_data, shuffle=True, n_node=n_node, n_price=n_price, n_category=n_category)
    model = trans_to_cuda(CoHHN(adjacency=train_data.adjacency,adjacency_pv=train_data.adjacency_pv,adjacency_vp=train_data.adjacency_vp,adjacency_pc=train_data.adjacency_pc,adjacency_cp=train_data.adjacency_cp,adjacency_cv=train_data.adjacency_cv,adjacency_vc=train_data.adjacency_vc,n_node=n_node,n_price=n_price,n_category = n_category,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset, num_heads=opt.num_heads))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()
