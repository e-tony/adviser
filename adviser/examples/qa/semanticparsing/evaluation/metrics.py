from sklearn.metrics import classification_report, f1_score
import numpy as np


def get_classification_report(y_true, pred, classes, output_dict=False):
    """Computes the classification report
    Args:
        y_true: list, a list of integers of true labels
        pred: list, a list of integers of predicted labels
        classes: list, a list of integers for all classes
        output_dict: bool, if True, the output will be a dictionary, else it will be the default format
    Returns:
        The sklearn classification report
    """
    return classification_report(y_true=y_true, y_pred=pred, labels=classes, output_dict=output_dict)


def get_macro_f1(y_true, pred):
    """Computes the macro f1 score
    Args:
        y_true: list, a list of integers of true labels
        pred: list, a list of integers of predicted labels
    Returns:
        A list of floats
    """
    return f1_score(y_true=y_true, y_pred=pred, average="macro")


def get_micro_f1(y_true, pred):
    """Computes the micro f1 score
    Args:
        y_true: list, a list of integers of true labels
        pred: list, a list of integers of predicted labels
    Returns:
        A list floats
    """
    return f1_score(y_true=y_true, y_pred=pred, average="micro")


def hits_at_k(ranks, k=10):
    """Computes the hits@k metric
    Args:
        ranks: list, a list of integers of rankings of predictions
    Returns:
        A list of floats
    """
    return float(len([r for r in ranks if r <= k])) / len(ranks)


def mrr(ranks):
    """Computes the mean reciprocal rank metric
    Args:
        ranks: list, a list of integers of rankings of predictions
    Returns:
        An array of floats
    """
    return np.average([1.0 / r for r in ranks])
