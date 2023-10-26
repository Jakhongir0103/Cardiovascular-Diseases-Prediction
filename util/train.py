import itertools
from typing import Dict, List

import numpy as np

from implementations import logistic_loss, logistic_loss_gradient, batch_iter


def reg_logistic_regression(
        tx_train: np.ndarray,
        y_train: np.ndarray,
        tx_valid: np.ndarray,
        y_valid: np.ndarray,
        lambda_: float,
        w: np.ndarray,
        max_iter: int,
        gamma: float,
        batch_size: int = 1,
        all_losses: bool = False,
        optimizer: str = "sgd"
):
    """
    Regularized logistic regression using SGD.
    Return the optimum w and corresponding train and validation loss.

    Args:
        tx_train:  shape=(N, D)
        y_train:  shape=(N, 1)
        tx_valid:  shape=(N, D)
        y_valid:  shape=(N, 1)
        lambda_: an array of lambdas
        w: numpy array, initial values for w
        max_iter: scalar
        gamma: an array of gammas
        batch_size: scalar
        all_losses: bool, if set to True returns an array of losses, otherwise returns the loss(scalar) of optimum w
        optimizer: either "sgd", "adagrad" or "adam"

    Returns:
        w: numpy array, optimal weights.
        train_losses: if all_losses is False -> scalar, train loss corresponding to the optimum w
                      if all_losses is True -> numpy array, train losses for each iteration
        valid_losses: if all_losses is False -> scalar, validation loss corresponding to the optimum w
                      if all_losses is True -> numpy array, validation losses for each iteration
    """
    assert optimizer in ["sgd", "adam", "adagrad"]

    if optimizer == "adagrad":
        gradients_sum = 0
    elif optimizer == "adam":
        gradients_sum = np.zeros(shape=(tx_train.shape[1], 1))

    if all_losses:
        train_losses = [logistic_loss(y_train, tx_train, w, lambda_)]
        valid_losses = [logistic_loss(y_valid, tx_valid, w, lambda_)]


    best_w = w
    lowest_loss = np.inf

    m = np.zeros(shape=(tx_train.shape[1], 1))
    v = np.zeros(shape=(tx_train.shape[1], 1))

    for it in range(max_iter):
        if it % 200 == 0:
            print(f"Iteration {it}/{max_iter} -> lowest loss {lowest_loss}")

        for batch_y, batch_x in batch_iter(y_train, tx_train, batch_size, shuffle=True):
            gradient = logistic_loss_gradient(batch_y, batch_x, w, lambda_)

            if optimizer == "sgd":
                w = w - gamma * gradient
            elif optimizer == "adagrad":
                gradients_sum += np.dot(gradient, gradient)
                h = np.sqrt(gradients_sum)  # h is scalar
                w = w - (1 / h) * gamma * gradient
            elif optimizer == "adam":
                beta_1 = 0.9
                beta_2 = 0.999
                m = beta_1 * m + (1 - beta_1) * gradient.reshape(-1, 1)
                v = beta_2 * v + (1 - beta_2) * (gradient ** 2).reshape(-1, 1)
                w = w.reshape(-1, 1) - (v ** 0.5) * gamma * m

                w.reshape((-1))

            # do not include the regularization term
            valid_loss = logistic_loss(y_valid, tx_valid, w, lambda_=None)
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_w = w

        if all_losses:
            train_losses.append(logistic_loss(y_train, tx_train, w, lambda_=None))
            valid_losses.append(logistic_loss(y_valid, tx_valid, w, lambda_=None))

    if not all_losses:
        train_losses = logistic_loss(y_train, tx_train, best_w, lambda_=None)
        valid_losses = lowest_loss

    return best_w, train_losses, valid_losses


def reg_logistic_regression_hyperparameters(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        hyperpar_grid: Dict[str, List],
        max_iter: int = 5000,
) -> List[Dict]:
    """
    GridSearch for regularized logistic regression.
    Return the optimum w and corresponding train loss, validation loss, lambda, gamma.

    Args:
        x_train:  shape=(N, D)
        y_train:  shape=(N, 1)
        x_valid:  shape=(N, D)
        y_valid:  shape=(N, 1)
        hyperpar_grid: Dict which contains keys "lambda_", "gamma", "batch_size", "optimizer" and whose values
        are lists of possible values for the hyperparameter
        max_iter: scalar
    :return: List of Dict containing hyperparameter settings and results
    """

    def cartesian_product(hyperpar_dict: Dict[str, List]):
        return [dict(zip(hyperpar_dict.keys(), values)) for values in itertools.product(*hyperpar_dict.values())]

    hyp_params_settings: List = cartesian_product(hyperpar_dict=hyperpar_grid)
    for hyp_params in hyp_params_settings:
        assert "lambda_" in hyp_params
        assert "gamma" in hyp_params
        assert "batch_size" in hyp_params
        assert "optimizer" in hyp_params

        w, train_loss, valid_loss = reg_logistic_regression(
            tx_train=x_train,
            y_train=y_train,
            tx_valid=x_valid,
            y_valid=y_valid,
            lambda_=hyp_params["lambda_"],
            max_iter=max_iter,
            gamma=hyp_params["gamma"],
            batch_size=hyp_params["batch_size"],
            w=np.random.random(size=x_train.shape[1]),
            optimizer=hyp_params["optimizer"],
            all_losses=False
        )
        hyp_params["best_weights"] = w
        hyp_params["train_loss"] = train_loss
        hyp_params["valid_loss"] = valid_loss

    return hyp_params_settings
