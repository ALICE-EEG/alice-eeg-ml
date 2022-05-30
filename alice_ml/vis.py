import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve, auc

def plot_aggregated_roc_curve(y_true, y_proba, ax):
    fprs, tprs = [], []
    for true, proba in zip(y_true, y_proba):
        fpr, tpr, _ = roc_curve(true, proba, drop_intermediate=False)
        fprs.extend(fpr)
        tprs.extend(tpr)

    points = pd.DataFrame({'fpr': fprs, 'tpr': tprs})
    mean_values = points.groupby('fpr').mean()
    value_stds = points.groupby('fpr').std()['tpr']

    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=18)

    ax.yaxis.tick_right()

    ax.tick_params(axis='both', which='both', length=0)
    ax.grid()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.plot([0, 1], [0, 1], lw=2, color='navy', linestyle='--')
    ax.plot(mean_values.index, mean_values['tpr'], lw=2, color='darkorange')
    ax.fill_between(mean_values.index, mean_values['tpr'] - 1.96 * value_stds,
                    mean_values['tpr'] + 1.96 * value_stds, alpha=0.2, lw=2, color='darkorange')


def plot_aggregated_pr_curve(y_true, y_proba, ax):
    precs, recs = [], []
    for true, proba in zip(y_true, y_proba):
        prec, rec, _ = precision_recall_curve(true, proba)
        precs.extend(prec)
        recs.extend(rec)

    points = pd.DataFrame({'prec': precs, 'rec': recs})
    mean_values = points.groupby('rec').mean()
    value_stds = points.groupby('rec').std()['prec']

    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=18)

    ax.yaxis.tick_right()

    ax.tick_params(axis='both', which='both', length=0)
    ax.grid()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y_true.mean(), 0, 1, lw=2, color='navy', linestyle='--')
    ax.plot(mean_values.index, mean_values['prec'], lw=2, color='darkorange')
    ax.fill_between(mean_values.index, mean_values['prec'] - 1.96 * value_stds,
                    mean_values['prec'] + 1.96 * value_stds, alpha=0.2, lw=2, color='darkorange')


