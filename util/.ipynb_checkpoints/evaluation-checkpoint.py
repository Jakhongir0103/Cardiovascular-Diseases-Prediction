import numpy as np

def accuracy(true_labels: np.ndarray, 
             predicted_labels: np.ndarray) -> float:
    """
    Calculate the accuracy of a classifier.
    :param true_labels: True class labels.
    :param predicted_labels: Predicted class labels.
    :return float: Accuracy of the classifier.
    """
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_samples = true_labels.shape[0]
    return correct_predictions / total_samples


def precision(true_labels: np.ndarray,
              predicted_labels: np.ndarray) -> float:
    """
    Calculate the precision of a classifier.
    :param true_labels: True class labels.
    :param predicted_labels: Predicted class labels.
    :return float: precision of the classifier.
    """
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))

    return true_positives / (true_positives + false_positives)


def recall(true_labels: np.ndarray,
           predicted_labels: np.ndarray) -> float:
    """
    Calculate the recall of a classifier.
    :param true_labels: True class labels.
    :param predicted_labels: Predicted class labels.
    :return float: recall of the classifier.
    """
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))

    return true_positives / (true_positives + false_negatives)


def f1_score(true_labels: np.ndarray,
             predicted_labels: np.ndarray) -> float:
    """
    Calculate the F1 score of a classifier.
    :param true_labels: True class labels.
    :param predicted_labels: Predicted class labels.
    :return float: F1 score of the classifier.
    """
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("Input arrays must have the same shape.")

    prec = precision(true_labels, predicted_labels)
    rec = recall(true_labels, predicted_labels)

    return 2 * (prec * rec) / (prec + rec)


def evaluation_summary(true_labels: np.ndarray,
                       predicted_labels: np.ndarray):
    print(f"Accuracy: {accuracy(true_labels, predicted_labels):.2%}")
    print(f"Precision: {precision(true_labels, predicted_labels):.2%}")
    print(f"Recall: {recall(true_labels, predicted_labels):.2%}")
    print(f"F1-score: {f1_score(true_labels, predicted_labels):.2%}")