from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate(model, X_val, y_val):
    preds = model.predict_proba(X_val)[:,1]

    roc = roc_auc_score(y_val, preds)

    precision, recall, _ = precision_recall_curve(y_val, preds)
    pr_auc = auc(recall, precision)

    return roc, pr_auc
