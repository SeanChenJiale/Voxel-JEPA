import numpy as np

def calculate_f1_from_confusion_matrix(confusion_matrix):
    """
    Calculate F1 scores (per class, macro, micro, weighted) from a confusion matrix.

    Args:
        confusion_matrix (np.ndarray): A square confusion matrix (e.g., 3x3 for 3 classes).

    Returns:
        dict: F1 scores for each class, macro F1, micro F1, and weighted F1.
    """
    num_classes = confusion_matrix.shape[0]
    f1_scores = []
    precision_list = []
    recall_list = []
    support = confusion_matrix.sum(axis=1)  # Number of true instances per class

    for i in range(num_classes):
        tp = confusion_matrix[i, i]  # True positives for class i
        fp = confusion_matrix[:, i].sum() - tp  # False positives for class i
        fn = confusion_matrix[i, :].sum() - tp  # False negatives for class i

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

    # Macro F1: Average of F1 scores
    macro_f1 = np.mean(f1_scores)

    # Micro F1: Global TP, FP, FN
    tp_micro = np.trace(confusion_matrix)
    fp_micro = confusion_matrix.sum() - tp_micro
    fn_micro = confusion_matrix.sum() - tp_micro
    micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Weighted F1: Weighted average of F1 scores
    total_support = support.sum()
    weighted_f1 = np.sum((support / total_support) * np.array(f1_scores))

    return {
        "f1_per_class": f1_scores,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1
    }

# Example usage
if __name__ == "__main__":
    # Example 3x3 confusion matrix
    confusion_matrix = np.array([
        [131, 42,11],
        [26, 37, 21],
        [40, 10, 39]
    ])

    f1_scores = calculate_f1_from_confusion_matrix(confusion_matrix)
    print("F1 Scores per Class:", f1_scores["f1_per_class"])
    print("Macro F1:", f1_scores["macro_f1"])
    print("Micro F1:", f1_scores["micro_f1"])
    print("Weighted F1:", f1_scores["weighted_f1"])

