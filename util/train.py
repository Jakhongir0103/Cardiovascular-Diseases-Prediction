import numpy as np
from tqdm import tqdm

from implementations import penalized_logistic_regression
from implementations import logistic_loss, logistic_loss_gradient, batch_iter


def reg_logistic_regression(tx_train: np.ndarray, y_train: np.ndarray,
                            tx_valid: np.ndarray, y_valid: np.ndarray,
                            lambda_: float, w: np.ndarray, max_iter: int, gamma: float, batch_size: int = 100):
    # init parameter
    # threshold = 1e-8
    train_losses = [logistic_loss(y_train, tx_train, w, lambda_)]
    valid_losses = [logistic_loss(y_valid, tx_valid, w, lambda_)]
    # start the logistic regression
    for it in tqdm(range(max_iter)):

        for batch_y, batch_x in batch_iter(y_train, tx_train, batch_size, shuffle=True):
            # USELESS train_loss = logistic_loss(batch_y, batch_x, w, lambda_)
            gradient = logistic_loss_gradient(batch_y, batch_x, w, lambda_)
            w = w - gamma * gradient

        train_losses.append(logistic_loss(y_train, tx_train, w, lambda_))
        valid_losses.append(logistic_loss(y_valid, tx_valid, w, lambda_))

    return w, train_losses, valid_losses
