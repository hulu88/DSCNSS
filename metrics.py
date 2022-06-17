import numpy as np
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

import warnings
warnings.filterwarnings('ignore')

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)

    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')

    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')

    return acc, f1_macro, precision_macro, recall_macro


def eval(y_true, y_pred, epoch=None):
    try:
        acc, f1, precision, recall = cluster_acc(y_true, y_pred)
        nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
        ari = ari_score(y_true, y_pred)
    except:  # 以防止 TypeError: cannot unpack non-iterable NoneType object
        acc, f1, precision, recall, nmi, ari = -1, -1, -1, -1, -1, -1
    # print(epoch, ': acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #       ', f1 {:.4f}'.format(f1), ', precision {:.4f}'.format(precision), ', recall {:.4f}'.format(recall))
    return {'acc': np.round(acc, 4), 'nmi': np.round(nmi, 4), 'ari': np.round(ari, 4),
            'f1': np.round(f1, 4), 'precision': np.round(precision, 4), 'recall': np.round(recall, 4)}
    # return {'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1, 'precision': precision, 'recall': recall}


'''
使用方法：


from metrics import eval  # 加载评估指标
......
result = eval(y_true, y_pred)  # 输出聚类指标的结果
print('acc {:.4f}, nmi {:.4f}, ari {:.4f}, f1 {:.4f}'.format(result['acc'], result['nmi'], result['ari'], result['f1']))

'''
