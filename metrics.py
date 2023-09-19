import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def calc_precision_recall(y, pred, top_k=slice(None)):
    y = y[top_k]
    pred = pred[top_k]

    mean = lambda x: sum(x)/len(x)
    precision, recall = [], []

    for y, y_pred in zip(y, pred):
        if len(y_pred):
            inter = set(y).intersection(y_pred)
            precision.append(len(inter)/len(y_pred))
            recall.append(len(inter)/len(y))
        else:
            precision.append(0)
            recall.append(0)

    return mean(precision), mean(recall)

def dcg(gains, top_k=slice(None)):
    gains = gains[top_k]

    denominator = np.array([i + 2 for i in range(len(gains))])
    denominator = np.log2(denominator)

    return (gains / denominator).sum()

def norm_dcg(ideal_list, predicted_list, top_k=slice(None)):
    gains = np.isin(predicted_list, ideal_list)
    pred_dcg = dcg(gains, top_k)
    ideal_dcg = dcg(np.ones(len(ideal_list)), top_k)

    return pred_dcg / ideal_dcg


def calc_ndcg(y_true, y_pred, top_k=slice(None)):
    ndcg_list = [norm_dcg(y, y_hat, top_k) for y,y_hat in zip(y_true, y_pred)]
    return np.array(ndcg_list).mean()
