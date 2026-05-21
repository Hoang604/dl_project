import numpy as np
import torch

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, roc_auc_score


def stas_datasets(n_pos, y_pred, y_true, y_score):
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    ok_pred = (y_pred == y_true).astype(int)
    # accuracy
    tot_pred_ok = ok_pred.sum()
    accuracy = tot_pred_ok / n_pos
    # confidence
    mean_score_ok = (ok_pred * y_score).sum() / \
        tot_pred_ok if tot_pred_ok > 0 else 0.0
    mean_score_wrong = ((1-ok_pred) * y_score).sum() / \
        (n_pos - tot_pred_ok) if (n_pos - tot_pred_ok) > 0 else 0.0
    # confusion matrix
    conf = confusion_matrix(y_true, y_pred)
    return accuracy, mean_score_ok, mean_score_wrong, conf


def compute_metrics(y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos,
                    y_score_neg,  y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg,
                    class_dict, target_far=0.05, verbose=True):

    # identify the index of the _unnown_ (negative) class in the output vectors
    if '_unknown_' in class_dict.keys():
        unk_idx = class_dict['_unknown_']
    else:
        unk_idx = -1

    '''
        Compute Accuracy Positive Set 
    '''
    n_pos = len(y_score_pos)
    y_score_pos = np.array(y_score_pos)
    y_pred_pos = np.array(y_pred_pos)
    y_true_pos = np.array(y_true_pos)
    accuracy_pos, mean_score_ok, mean_score_wrong, conf = \
        stas_datasets(n_pos, y_pred_pos, y_true_pos, y_score_pos)
    print('Accuracy Positive:', accuracy_pos)
    print('Avg score: ', mean_score_ok, mean_score_wrong)
    print(conf)

    if y_score_neg is None:
        y_score_neg = []

    n_neg = len(y_score_neg)
    if n_neg > 0:
        accuracy_neg, mean_score_ok, mean_score_wrong, conf = \
            stas_datasets(n_neg, y_pred_neg, y_true_neg, y_score_neg)
        print('Accuracy Negative: ', accuracy_neg)
        print('Avg score: ', mean_score_ok, mean_score_wrong)
        print(conf)
    else:
        accuracy_neg = 0.0

    auroc = 0.0
    acc_xxfar = thr_xxfar = frr_xxfar = cerr_xxfar = 0.0

    if n_neg > 0:
        y = np.array([1] * n_pos + [-1] * n_neg)

        y_pred_close = y_pred_close_pos + y_pred_close_neg
        y_pred_close = torch.Tensor(y_pred_close)
        y_pred_close, _ = y_pred_close.max(dim=1)

        y_score = np.array(y_pred_close)
        fpr, tpr, thresholds = roc_curve(y, y_score)
        auroc = roc_auc_score(y, y_score)
        print('auroc: ', auroc)

        # search thr at target_far FAR
        print(fpr, tpr, thresholds)
        th_idx = -1
        for i, item in enumerate(fpr):
            if item > target_far:
                print(item, i, tpr[th_idx])
                break
            else:
                th_idx = i

        # compute accuracy and FRR at target FAR
        if th_idx != -1:
            thr_xxfar = thresholds[th_idx].astype(np.float64)
            mask = y_score_pos < thr_xxfar
            y_pred_pos[mask] = unk_idx
            frr_xxfar = (y_pred_pos == unk_idx).sum() / n_pos
            print('after masking:', (y_pred_pos == unk_idx).sum(), frr_xxfar)
            acc_xxfar, _, _, conf = \
                stas_datasets(n_pos, y_pred_pos, y_true_pos, y_score_pos)
            print(conf)

        print('Accuracy Positive:', acc_xxfar)
        print('THR Positive:', thr_xxfar)

    return {'aucROC': auroc, 'n_pos': n_pos, 'n_neg': n_neg,
            'accuracy_pos': accuracy_pos,
            'accuracy_neg': accuracy_neg,
            'acc_prec95': acc_xxfar, 'thr_prec95': thr_xxfar,
            'frr_prec95': frr_xxfar, 'cerr_prec95': cerr_xxfar}
