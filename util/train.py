import numpy as np
from tqdm import tqdm

from implementations import penalized_logistic_regression
from implementations import logistic_loss, logistic_loss_gradient, batch_iter


def reg_logistic_regression(tx_train: np.ndarray, y_train: np.ndarray,
                            tx_valid: np.ndarray, y_valid: np.ndarray,
                            lambda_: float, w: np.ndarray, max_iter: int, gamma: float, batch_size: int = 100, all_losses: bool = False):
    if all_losses:
        train_losses = [logistic_loss(y_train, tx_train, w, lambda_)]
        valid_losses = [logistic_loss(y_valid, tx_valid, w, lambda_)]

    # choose the iterator
    if all_losses:
        loop_iterator = tqdm(range(max_iter))
    else:
        loop_iterator = range(max_iter)

    for _ in loop_iterator:
    # for it in tqdm(range(max_iter)):

        for batch_y, batch_x in batch_iter(y_train, tx_train, batch_size, shuffle=True):
            # USELESS train_loss = logistic_loss(batch_y, batch_x, w, lambda_)
            gradient = logistic_loss_gradient(batch_y, batch_x, w, lambda_)
            w = w - gamma * gradient
        if all_losses: 
            train_losses.append(logistic_loss(y_train, tx_train, w, lambda_))
            valid_losses.append(logistic_loss(y_valid, tx_valid, w, lambda_))
    if not all_losses:    
        train_losses = logistic_loss(y_train, tx_train, w, lambda_)
        valid_losses = logistic_loss(y_valid, tx_valid, w, lambda_)

    return w, train_losses, valid_losses

def reg_logistic_regression_hyperparameters(x_train, y_train, x_valid, y_valid, lambdas, gammas, w_initials, max_iter=5000, batch_size=100):
    best_w = None
    best_valid_loss = float('inf')
    best_lambda = None
    best_gamma = None
    best_w_value = None
    train_losses_all = []
    valid_losses_all = []
    w_all = []

    for lambda_ in tqdm(lambdas):
    # for lambda_ in lambdas:
        for gamma in gammas:
            for w_value in w_initials:
                # Perform logistic regression with the current hyperparameters
                w, train_losses, valid_losses = reg_logistic_regression_here(
                    tx_train=x_train, 
                    y_train=y_train, 
                    tx_valid=x_valid, 
                    y_valid=y_valid, 
                    lambda_=lambda_, 
                    max_iter=max_iter, 
                    gamma=gamma, 
                    batch_size=batch_size, 
                    w=w_value
                )
                w_all.append(w)
                train_losses_all.append(train_losses)
                valid_losses_all.append(valid_losses)

    # Find the index of the minimum valid loss
    best_loss_index = np.argmin(valid_losses_all)
    # Get the corresponding indices for lambda, gamma, and w_initial
    best_lambda_index, best_gamma_index, best_w_initial_index = np.unravel_index(best_loss_index, (len(lambdas), len(gammas), len(w_initials)))

    # Retrieve the corresponding hyperparameters
    best_lambda = lambdas[best_lambda_index]
    best_gamma = gammas[best_gamma_index]
    best_w_initial = w_initials[best_w_initial_index]

    # Get the best weights and losses
    best_w = w_all[best_loss_index]
    best_train_losses = train_losses_all[best_loss_index]
    best_valid_losses = valid_losses_all[best_loss_index]

    return best_w, best_train_losses, best_valid_losses, best_lambda, best_gamma, best_w_initial