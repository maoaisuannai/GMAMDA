import timeit
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.GMAMDA import GMAMDA
from metric import *
from sklearn.model_selection import KFold

device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--nhid', type=int, default=256, help='gcn_dim')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='metadis', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--negative', default='ANHS', type=str, help='the method of neagative sampling [random,ANHS]')
    parser.add_argument('--positive_ratio', type=float, default=1.0, help='Percentage of positive samples used')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/GMAMDA/'


    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']

    if args.negative == 'random':
        data = data_processing_random(data, args)
    else:
        data = data_processing_neg(data, args)

    adj_mat = construct_adj_mat(data['adj'])

    het_mat1 = np.vstack((np.hstack((data['drs1'], data['adj'])), np.hstack((data['adj'].T, data['dis1']))))
    adj_edge_index1 = get_edge_index(het_mat1).to(device)
    het_mat1 = torch.tensor(het_mat1, dtype=torch.float32, device=device)

    het_mat2 = np.vstack((np.hstack((data['drs2'], data['adj'])), np.hstack((data['adj'].T, data['dis2']))))
    adj_edge_index2 = get_edge_index(het_mat2).to(device)
    het_mat2 = torch.tensor(het_mat1, dtype=torch.float32, device=device)

    het_mat3 = np.vstack((np.hstack((data['drs3'], data['adj'])), np.hstack((data['adj'].T, data['dis3']))))
    adj_edge_index3 = get_edge_index(het_mat3).to(device)
    het_mat3 = torch.tensor(het_mat1, dtype=torch.float32, device=device)

    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

    fold_best_results = []
    AUCs, AUPRs = [], []
    accuracies, precisions, recalls = [], [], []
    f1s, mccs = [], []
    spes = []
    all_results = []
    fold_results = []

    print('Dataset:', args.dataset)

    samples = data['all_drdi']
    labels = data['all_label']

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.random_seed)

    all_results = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(samples)):
        print(f'\nTraining Fold {fold_idx + 1}/{args.k_fold}')

        visualization_epochs = [0, 99, 999]  # For epochs 1, 50, 100
        save_dir = os.path.join('visualization_para/3', f'fold_{fold_idx + 1}')
        os.makedirs(save_dir, exist_ok=True)

        X_train = torch.LongTensor(samples[train_index]).to(device)
        Y_train = torch.LongTensor(labels[train_index]).to(device)
        X_test = torch.LongTensor(samples[val_index]).to(device)
        Y_test = torch.LongTensor(labels[val_index]).to(device)
        Y_test = Y_test.cpu().numpy()

        edge_idx_train1 = adj_edge_index1[:, train_index].to(device)
        edge_idx_test1 = adj_edge_index1[:, val_index].to(device)

        edge_idx_train2 = adj_edge_index2[:, train_index].to(device)
        edge_idx_test2 = adj_edge_index2[:, val_index].to(device)

        edge_idx_train3 = adj_edge_index3[:, train_index].to(device)
        edge_idx_test3 = adj_edge_index3[:, val_index].to(device)

        print(Metric)

        model = GMAMDA(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc, best_spe = 0, 0, 0, 0, 0, 0, 0, 0

        for epoch in range(args.epochs):
            fold_results2 = []  # 存储当前fold的所有结果
            model.train()
            _, train_score = model(X_train, adj_mat, edge_idx_train1, edge_idx_train2, edge_idx_train3)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score = model(X_test, adj_mat, edge_idx_test1, edge_idx_test2, edge_idx_test3)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc, spe = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                    round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]

            fold_results2.append('\t\t'.join(map(str, show)))
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc, best_spe = AUPR, accuracy, precision, recall, f1, mcc, spe
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)
                best_labels = Y_test
                best_scores = test_prob

        # 保存每个fold的结果
        fold_results.append((best_labels, best_scores))
        # 收集统计数据
        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        accuracies.append(best_accuracy)
        precisions.append(best_precision)
        recalls.append(best_recall)
        f1s.append(best_f1)
        mccs.append(best_mcc)
        spes.append(best_spe)

    # 计算平均值和标准差
    metrics = {
        'AUC': (AUCs, np.mean(AUCs), np.std(AUCs)),
        'AUPR': (AUPRs, np.mean(AUPRs), np.std(AUPRs)),
        'Accuracy': (accuracies, np.mean(accuracies), np.std(accuracies)),
        'Precision': (precisions, np.mean(precisions), np.std(precisions)),
        'Recall': (recalls, np.mean(recalls), np.std(recalls)),
        'F1': (f1s, np.mean(f1s), np.std(f1s)),
        'MCC': (mccs, np.mean(mccs), np.std(mccs)),
        'Spe': (spes, np.mean(spes), np.std(spes))
    }

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')


