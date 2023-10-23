import numpy as np

def accuracy(true_labels: np.ndarray, 
             predicted_labels: np.ndarray) -> float:
    """
    Calculate the accuracy of a classifier.
    :param true_labels (numpy.ndarray): True class labels.
    :param predicted_labels (numpy.ndarray): Predicted class labels.
    :return float: Accuracy of the classifier.
    """
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_samples = true_labels.shape[0]
    return correct_predictions / total_samples


def f1_score(true_labels: np.ndarray,
             predicted_labels: np.ndarray) -> float:
    """
    Calculate the F1 score of a classifier.
    :param true_labels (numpy.ndarray): True class labels.
    :param predicted_labels (numpy.ndarray): Predicted class labels.
    :return float: F1 score of the classifier.
    """
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    precision = precision(true_labels, predicted_labels)
    recall = recall(true_labels, predicted_labels)
    
    return 2 * (precision * recall) / (precision + recall)
    

def precision(true_labels: np.ndarray,
              predicted_labels: np.ndarray) -> float:
    """
    Calculate the precision of a classifier.
    :param true_labels (numpy.ndarray): True class labels.
    :param predicted_labels (numpy.ndarray): Predicted class labels.
    :return float: precision of the classifier.
    """
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positives = np.sum((true_labels == -1) & (predicted_labels == 1))

    return true_positives / (true_positives + false_positives)


def recall(true_labels: np.ndarray,
           predicted_labels: np.ndarray) -> float:
    """
    Calculate the recall of a classifier.
    :param true_labels (numpy.ndarray): True class labels.
    :param predicted_labels (numpy.ndarray): Predicted class labels.
    :return float: recall of the classifier.
    """
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == -1))

    return true_positives / (true_positives + false_negatives)
