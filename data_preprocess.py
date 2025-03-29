import os

import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.decomposition import PCA

device = torch.device('cuda')


def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()

    adj = pd.read_csv(args.data_dir + 'adj.csv', index_col=0).to_numpy()
    adj = reduce_adj_matrix(adj, args)

    data['drug_number'] = 2033
    data['disease_number'] = 216

    data['adj'] = adj

    one_indices = np.argwhere(adj == 1).astype(np.int32)
    data['drdi'] = one_indices

    return data


def data_processing_neg(data, args):

    drdi_matrix = torch.tensor(data['adj'])
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    train_pos_set = defaultdict(list)

    # 遍历邻接矩阵,提取正样本集
    for i in range(2033):
        for j in range(216):
            if drdi_matrix[i][j] > 0:
                train_pos_set[i].append(j)

    pos_index = torch.tensor(one_index, dtype=torch.int64)
    drug = pos_index[:, 0]
    pos_disease = pos_index[:, 1]

    candadite_dis = sampling(one_index, train_pos_set, 32)

    # -----------------------meta--dis---------------------------------
    meta_st = pd.read_csv('data/metadis/metabolite_structure_sim.csv', header=None, dtype=np.float32).to_numpy()
    meta_ie = pd.read_csv('data/metadis/metabolite_entropy_sim.csv', header=None, dtype=np.float32).to_numpy()
    meta_gip = pd.read_csv('data/metadis/metabolite_gip_sim.csv', header=None, dtype=np.float32).to_numpy()

    dis_se = pd.read_csv('data/metadis/disease_semantic_similarity.csv', header=None, dtype=np.float32).to_numpy()
    dis_ie = pd.read_csv('data/metadis/disease _information_entropy_similarity.csv', header=None,
                         dtype=np.float32).to_numpy()
    dis_gip = pd.read_csv('data/metadis/disease_GIP_similarity.csv', header=None, dtype=np.float32).to_numpy()

    meta_mean = (meta_st + meta_ie + meta_gip) / 3
    dis_mean = (dis_se + dis_ie + dis_gip) / 3
    drs = np.where(meta_st == 0, meta_gip, meta_mean)
    dis = np.where(dis_se == 0, dis_gip, dis_mean)


    pca = PCA(n_components=64)
    drug_emb = pca.fit_transform(drs)
    dis_emb = pca.fit_transform(dis)

    neg_dis = adaptive_negative_sampling(drug_emb, dis_emb, drug, candadite_dis, pos_disease, 0.1, 1.0, -2)

    neg_index = torch.stack([drug, neg_dis], dim=1)
    neg_index = neg_index.tolist()

    index = np.array(one_index + neg_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(neg_index), dtype=int)

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    data['drs1'] = meta_st
    data['drs2'] = meta_gip
    data['drs3'] = meta_ie
    data['dis1'] = dis_se
    data['dis2'] = dis_gip
    data['dis3'] = dis_ie
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p

    return data


def k_fold(data, args, edge_idx):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=args.random_seed, shuffle=True)
    X = data['all_drdi']
    Y = data['all_label']
    X_train_all, X_test_all, Y_train_all, Y_test_all, edge_idx_train_all, edge_idx_test_all = [], [], [], [], [], []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    for i in range(k):
        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1),
                                columns=['drug', 'disease', 'label'])
        X_train1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_train.csv')
        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1),
                               columns=['drug', 'disease', 'label'])
        X_test1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_test.csv')

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    data['edge_idx_train'] = edge_idx_train_all
    data['edge_idx_test'] = edge_idx_test_all
    return data


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])

    return drdr_graph, didi_graph, data


def sampling(user_item, train_set, n):
    n_items = 216
    user_item = torch.Tensor(user_item)
    neg_items = []
    for user, _ in user_item.cpu().numpy():
        user = int(user)
        negitems = []
        for i in range(n):
            while True:
                negitem = random.choice(range(n_items))
                if negitem not in train_set[user]:
                    break
            negitems.append(negitem)
        neg_items.append(negitems)
    return neg_items


def adaptive_negative_sampling(user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item, beta, alpha, p):
    batch_size = user.shape[0]
    s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]
    n_e = item_gcn_emb[neg_candidates]
    neg_candidates = torch.tensor(neg_candidates)
    s_e = torch.tensor(s_e)
    p_e = torch.tensor(p_e)
    n_e = torch.tensor(n_e)

    p_scores = ((s_e * p_e).sum(dim=-1)).unsqueeze(dim=1)  # [batch_size, 1]
    n_scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]

    scores = torch.abs(n_scores - beta * (p_scores + alpha).pow(p + 1))

    """adaptive negative sampling"""
    indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
    neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()

    return neg_item


def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    rna_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    dis_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((rna_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def data_processing_random(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    # -----------------------meta--dis---------------------------------
    meta_st = pd.read_csv('data/metadis/metabolite_structure_sim.csv', header=None, dtype=np.float32).to_numpy()
    meta_ie = pd.read_csv('data/metadis/metabolite_entropy_sim.csv', header=None, dtype=np.float32).to_numpy()
    meta_gip = pd.read_csv('data/metadis/metabolite_gip_sim.csv', header=None, dtype=np.float32).to_numpy()

    dis_se = pd.read_csv('data/metadis/disease_semantic_similarity.csv', header=None, dtype=np.float32).to_numpy()
    dis_ie = pd.read_csv('data/metadis/disease _information_entropy_similarity.csv', header=None,
                         dtype=np.float32).to_numpy()
    dis_gip = pd.read_csv('data/metadis/disease_GIP_similarity.csv', header=None, dtype=np.float32).to_numpy()

    meta_mean = (meta_st + meta_ie + meta_gip) / 3
    dis_mean = (dis_se + dis_ie + dis_gip) / 3
    drs = np.where(meta_st == 0, meta_gip, meta_mean)
    dis = np.where(dis_se == 0, dis_gip, dis_mean)

    data['drs1'] = meta_st
    data['drs2'] = meta_gip
    data['drs3'] = meta_ie
    data['dis1'] = dis_se
    data['dis2'] = dis_gip
    data['dis3'] = dis_ie

    data['drs'] = drs
    data['dis'] = dis
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p

    return data


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 0.3:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def reduce_adj_matrix(adj, args):
    one_indices = np.where(adj == 1)

    total_ones = len(one_indices[0])
    keep_num = int(total_ones * args.positive_ratio)
    keep_indices = np.random.choice(total_ones, keep_num, replace=False)
    reduced_adj = adj.copy()

    for i in range(total_ones):
        if i not in keep_indices:
            reduced_adj[one_indices[0][i], one_indices[1][i]] = 0

    return reduced_adj
