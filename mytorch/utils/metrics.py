
# ============================================================================
# FILE: mytorch/utils/metrics.py
# ============================================================================
"""Evaluation metrics"""

import numpy as np

def accuracy(predictions, targets):
    """Calculate accuracy"""
    pred_classes = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
    target_classes = targets if targets.ndim == 1 else np.argmax(targets, axis=-1)
    return (pred_classes == target_classes).mean()

def precision_recall_f1(predictions, targets, num_classes=None):
    """Calculate precision, recall, and F1 score"""
    pred_classes = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
    target_classes = targets if targets.ndim == 1 else np.argmax(targets, axis=-1)
    
    if num_classes is None:
        num_classes = max(pred_classes.max(), target_classes.max()) + 1
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for c in range(num_classes):
        tp = ((pred_classes == c) & (target_classes == c)).sum()
        fp = ((pred_classes == c) & (target_classes != c)).sum()
        fn = ((pred_classes != c) & (target_classes == c)).sum()
        
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0
    
    return precision.mean(), recall.mean(), f1.mean()

def confusion_matrix(predictions, targets, num_classes=None):
    """Calculate confusion matrix"""
    pred_classes = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
    target_classes = targets if targets.ndim == 1 else np.argmax(targets, axis=-1)
    
    if num_classes is None:
        num_classes = max(pred_classes.max(), target_classes.max()) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(target_classes, pred_classes):
        cm[t, p] += 1
    
    return cm