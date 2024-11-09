import numpy as np
from sklearn import metrics


def cal_auc(label_list, predict_ppl_list):
    fpr, tpr, thresholds = metrics.roc_curve(label_list, predict_ppl_list)
    auc = metrics.auc(fpr, tpr)
    return auc


def cal_acc(label_list, predict_ppl_list, eer_threshold=None):
    if eer_threshold is None:
        eer_threshold = cal_eer_threshold(label_list, predict_ppl_list)
    # print(eer_threshold)
    predict_list = [0 if x < eer_threshold else 1 for x in predict_ppl_list]
    acc = metrics.accuracy_score(label_list, predict_list)
    return acc


def cal_eer_threshold(label_list, predict_ppl_list):
    fpr, tpr, thresholds = metrics.roc_curve(label_list, predict_ppl_list)
    # Find the point where FPR and FNR are equal (EER point)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold


def cal_ppl_score(label_list, predict_ppl_list):
    label_list, predict_ppl_list = np.array(label_list), np.array(predict_ppl_list)
    label_0_ppl_list = predict_ppl_list[label_list == 0]
    label_1_ppl_list = predict_ppl_list[label_list == 1]
    pooled_std = np.sqrt((np.std(label_0_ppl_list, ddof=1) ** 2 + np.std(label_1_ppl_list, ddof=1) ** 2) / 2)
    return float((-np.mean(label_0_ppl_list) + np.mean(label_1_ppl_list)) / pooled_std)


def cal_ppl_diff(label_list, predict_ppl_list):
    label_list, predict_ppl_list = np.array(label_list), np.array(predict_ppl_list)
    label_0_ppl_list = predict_ppl_list[label_list == 0]
    label_1_ppl_list = predict_ppl_list[label_list == 1]
    return float(-np.mean(label_0_ppl_list) + np.mean(label_1_ppl_list))


def cal_all_metrics(label_list, predict_ppl_list):
    return {
        'ppl_score': cal_ppl_score(label_list, predict_ppl_list),
        'ppl_diff': cal_ppl_diff(label_list, predict_ppl_list),
        'acc': cal_acc(label_list, predict_ppl_list),
        'auc': cal_auc(label_list, predict_ppl_list),
    }

