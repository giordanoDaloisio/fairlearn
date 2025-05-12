import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from fairlearn.metrics import MetricFrame
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def _get_groups(data, label_name, positive_label, group_condition):
    query = "&".join([str(k) + "==" + str(v) for k, v in group_condition.items()])
    label_query = label_name + "==" + str(positive_label)
    unpriv_group = data.query(query)
    unpriv_group_pos = data.query(query + "&" + label_query)
    priv_group = data.query("~(" + query + ")")
    priv_group_pos = data.query("~(" + query + ")&" + label_query)
    return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos


def _compute_probs(data_pred, label_name, positive_label, group_condition):
    unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = _get_groups(
        data_pred, label_name, positive_label, group_condition
    )
    unpriv_group_prob = len(unpriv_group_pos) / len(unpriv_group)
    priv_group_prob = len(priv_group_pos) / len(priv_group)
    return unpriv_group_prob, priv_group_prob


def _compute_tpr_fpr(y_true, y_pred, positive_label):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == float(positive_label):
            if y_true[i] == y_pred[i]:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] == float(positive_label):
                FP += 1
            else:
                TN += 1
    if TP + FN == 0:
        TPR = 0
    else:
        TPR = TP / (TP + FN)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    return FPR, TPR

def _compute_fpr_fnr(y_true, y_pred, positive_label):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == float(positive_label):
            if y_true[i] == y_pred[i]:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] == float(positive_label):
                FP += 1
            else:
                TN += 1
    if TP + FN == 0:
        FNR = 0
    else:
        FNR = FN / (TP + FN)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    return FNR, FPR

def _compute_tpr_fpr_groups(data_pred, label, group_condition, positive_label):
    query = "&".join([f"{k}=={v}" for k, v in group_condition.items()])
    unpriv_group = data_pred.query(query)
    priv_group = data_pred.drop(unpriv_group.index)
    y_true_unpriv = unpriv_group["y_true"].values.ravel()
    y_pred_unpric = unpriv_group[label].values.ravel()
    y_true_priv = priv_group["y_true"].values.ravel()
    y_pred_priv = priv_group[label].values.ravel()
    fpr_unpriv, tpr_unpriv = _compute_tpr_fpr(
        y_true_unpriv, y_pred_unpric, positive_label
    )
    fpr_priv, tpr_priv = _compute_tpr_fpr(y_true_priv, y_pred_priv, positive_label)
    return fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv

def _compute_fnr_fpr_groups(data_pred, label, group_condition, positive_label):
    query = "&".join([f"{k}=={v}" for k, v in group_condition.items()])
    unpriv_group = data_pred.query(query)
    priv_group = data_pred.drop(unpriv_group.index)
    y_true_unpriv = unpriv_group["y_true"].values.ravel()
    y_pred_unpric = unpriv_group[label].values.ravel()
    y_true_priv = priv_group["y_true"].values.ravel()
    y_pred_priv = priv_group[label].values.ravel()
    fnr_unpriv, fpr_unpriv = _compute_fpr_fnr(
        y_true_unpriv, y_pred_unpric, positive_label
    )
    fnr_priv, fpr_priv = _compute_fpr_fnr(y_true_priv, y_pred_priv, positive_label)
    return fnr_unpriv, fpr_unpriv, fnr_priv, fpr_priv

def disparate_impact(data_pred, group_condition, label_name, positive_label):
    """
    Computes the disparate impact metric.
    Args:
        data_pred (pd.DataFrame): DataFrame containing the predictions and true labels.
        group_condition (dict): Dictionary specifying the group condition.
        label_name (str): Name of the predicted label column.
        positive_label: The positive label value.
    Returns:
        float: Disparate impact value.
    """
    
    unpriv_group_prob, priv_group_prob = _compute_probs(
        data_pred, label_name, positive_label, group_condition
    )
    return (
        min(unpriv_group_prob / priv_group_prob, priv_group_prob / unpriv_group_prob)
        if unpriv_group_prob != 0 and priv_group_prob != 0
        else 0
    )


def statistical_parity(
    data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str
):
    
    """
    Computes the statistical parity metric.
    Args:
        data_pred (pd.DataFrame): DataFrame containing the predictions and true labels.
        group_condition (dict): Dictionary specifying the group condition.
        label_name (str): Name of the predicted label column.
        positive_label: The positive label value.
    Returns:
        float: Disparate impact value.
    """

    query = "&".join([f"{k}=={v}" for k, v in group_condition.items()])
    label_query = label_name + "==" + str(positive_label)
    unpriv_group_prob = len(data_pred.query(query + "&" + label_query)) / len(
        data_pred.query(query)
    )
    priv_group_prob = len(data_pred.query("~(" + query + ")&" + label_query)) / len(
        data_pred.query("~(" + query + ")")
    )
    return unpriv_group_prob - priv_group_prob


def average_odds_difference(
    data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str
):
    fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = _compute_tpr_fpr_groups(
        data_pred, label_name, group_condition, positive_label
    )
    return ((tpr_priv - tpr_unpriv) + (fpr_priv - fpr_unpriv)) / 2


def equalized_odds(
    data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str
):
    fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = _compute_tpr_fpr_groups(
        data_pred, label_name, group_condition, positive_label
    )
    return tpr_unpriv - tpr_priv


def true_pos_diff(
    data_pred: pd.DataFrame, group_condition: str, label: str, positive_label: int
):
    fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = _compute_tpr_fpr_groups(
        data_pred, label, group_condition, positive_label
    )
    return fpr_unpriv - fpr_priv


def false_pos_diff(
    data_pred: pd.DataFrame, group_condition: str, label: str, positive_label: int
):
    fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = _compute_tpr_fpr_groups(
        data_pred, label, group_condition, positive_label
    )
    return fpr_unpriv - fpr_priv


def zero_one_loss_diff(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: list
):
    mf = MetricFrame(
        metrics=zero_one_loss,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return mf.difference()


def accuracy(df_pred: pd.DataFrame, label: str):
    return accuracy_score(df_pred["y_true"].values, df_pred[label].values)


def precision(df_pred: pd.DataFrame, label: str):
    return precision_score(
        df_pred["y_true"].values, df_pred[label].values, average="weighted"
    )


def recall(df_pred: pd.DataFrame, label: str):
    return recall_score(
        df_pred["y_true"].values, df_pred[label].values, average="weighted"
    )


def f1(df_pred: pd.DataFrame, label: str):
    return f1_score(df_pred["y_true"].values, df_pred[label].values, average="weighted")


def auc(df_pred: pd.DataFrame, label: str):
    return roc_auc_score(df_pred["y_true"].values, df_pred[label].values)


def euclidean_distance(df_pred, label):
    return np.linalg.norm(df_pred["y_true"].values - df_pred[label].values) / len(
        df_pred
    )


def manhattan_distance(df_pred, label):
    return np.sum(np.abs(df_pred["y_true"].values - df_pred[label].values)) / len(
        df_pred
    )


def mahalanobis_distance(df_pred, label):
    return np.sqrt(
        np.sum(np.square(df_pred["y_true"].values - df_pred[label].values))
    ) / len(df_pred)


def norm_data(data):
    return abs(1 - abs(data))
