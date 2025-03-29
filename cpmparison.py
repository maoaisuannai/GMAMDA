import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc


def load_fold_results(model_name):
    return np.load(f'all/{model_name}/fold_results.npy', allow_pickle=True)


def plot_combined_pr_curves(root_path, model_names):
    # plt.figure(figsize=(12, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 为每个模型定义颜色
    colors = {
        'KATZ': '#6F6F6F',
        'RWR': '#397FC7',
        'GCNAT': '#32037D',
        'GRGMF': '#F0A73A',
        'MDA-AENMF': '#A4E048',
        'GMAMDA': '#F94141'  # 或者使用 'orangered' 如果你想要橙红色
    }

    for model_name in model_names:
        try:
            fold_results = load_fold_results(model_name)
            mean_recall = np.linspace(0, 1, 8392)
            mean_fpr = np.linspace(0, 1, 8392)
            precisions = []
            pr_aucs = []
            tprs = []
            roc_aucs = []

            for y_true, y_pred_proba in fold_results:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
                pr_auc = average_precision_score(y_true, y_pred_proba)
                pr_aucs.append(pr_auc)
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                roc_aucs.append(roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_roc_auc = auc(mean_fpr, mean_tpr)
            std_roc_auc = np.std(roc_aucs)

            mean_precision = np.mean(precisions, axis=0)
            mean_pr_auc = np.mean(pr_aucs)
            std_pr_auc = np.std(pr_aucs)

            ax1.plot(mean_fpr, mean_tpr, color=colors[model_name], lw=2,
                     label=f'{model_name} (AUC = {round(mean_roc_auc, 4):.4f} ± {round(std_roc_auc, 4):.4f})')

            ax2.plot(mean_recall, mean_precision, color=colors[model_name], lw=2,
                     label=f'{model_name} (AUPR = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})')
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")

    # ax1.plot([0, 1], [0, 1], linestyle=':', lw=2, color='r')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")


    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve Comparison')
    ax2.legend(loc="lower left")
    ax2.grid(True)

    plt.tight_layout()

    save_path = os.path.join(root_path, 'combined_pr_curves3.png')
    plt.savefig(save_path)
    plt.close()


# 定义模型名称
model_names = ['KATZ', 'RWR', 'GCNAT', 'GRGMF', 'MDA-AENMF', 'GMAMDA']

# 创建保存结果的目录
root_path = os.path.join('results2', 'PR_comparison')
os.makedirs(root_path, exist_ok=True)

# 绘制组合的PR曲线
plot_combined_pr_curves(root_path, model_names)