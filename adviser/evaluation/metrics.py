from sklearn.metrics import classification_report, f1_score


def print_classification_report(gold, pred, classes):
    report = classification_report(y_true=gold, y_pred=pred, target_names=classes)
    print(report)
    return report


def get_macro_f1(gold, pred):
    return f1_score(y_true=gold, y_pred=pred, average="macro")


def get_micro_f1(gold, pred):
    return f1_score(y_true=gold, y_pred=pred, average="micro")
