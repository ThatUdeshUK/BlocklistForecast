import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def plot_history(history, title=None):
    plt.figure(figsize=(10.5, 7))
    history_df = pd.DataFrame(history, columns=['Loss']).reset_index().rename(columns={'index': 'Epoch'})
    sns.lineplot(x='Epoch', y='Loss', data=history_df, linewidth=5).set(title="" if title is None else title)
    plt.show()


def score(labels, pred, proba=None, multi_class=False):
    accuracy = accuracy_score(labels, pred)
    
    avg = 'weighted' if multi_class else 'binary'
    f1 = f1_score(labels, pred, average=avg)
    prec, recall = precision_score(labels, pred, average=avg), recall_score(labels, pred, average=avg)
    
    fpr = None
    if not multi_class:
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()    
        fpr = fp/(fp+tn)
    
    scores = {'acc': accuracy, 'f1': f1, 'prec': prec, 'recall': recall, 'fpr': fpr}
    
    if proba is not None:
        scores['auc'] = roc_auc_score(labels, proba, average='weighted', multi_class='ovr')
    
    return scores
