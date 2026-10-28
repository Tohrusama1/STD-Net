"""Binary timestamp metrics; a single-class AUC is undefined (JSON null)."""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, log_loss, classification_report)
import numpy as np


def classification_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities).ravel()
    predictions = probabilities >= threshold
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        'brier_score': float(brier_score_loss(labels, probabilities)),
        'log_loss': float(log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        'threshold': threshold,
        'num_timestamps': int(labels.size),
        'per_class': classification_report(labels, predictions, labels=[0, 1],
                                            target_names=['Normal', 'Attack'], output_dict=True,
                                            zero_division=0),
    }
