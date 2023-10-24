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
    false_positives = np.sum((true_labels == -1) & (predicted_labels == 1))

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
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == -1))

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


def confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray):
    # Calculate the confusion matrix
    classes = [1, -1]
    cm = np.zeros((len(classes), len(classes)), dtype=float)

    for i_idx, i in enumerate(classes):
        for j_idx, j in enumerate(classes):
            cm[i_idx, j_idx] = 100 * np.sum((true_labels == i) & (predicted_labels == j)) / len(true_labels)
            
    # Create the confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    cax = plt.matshow(cm, cmap='Blues')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f'{cm[i, j]:.2f}%', va='center', ha='center', color='black', fontsize=12)

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()