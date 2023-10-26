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

    Returns:
        w: numpy array, optimal weights.
        train_losses: if all_losses is False -> scalar, train loss corresponding to the optimum w
                      if all_losses is True -> numpy array, train losses for each iteration
        valid_losses: if all_losses is False -> scalar, validation loss corresponding to the optimum w
                      if all_losses is True -> numpy array, validation losses for each iteration
    """
    if all_losses:
        train_losses = [logistic_loss(y_train, tx_train, w, lambda_)]
        valid_losses = [logistic_loss(y_valid, tx_valid, w, lambda_)]

    for _ in range(max_iter):
        # if _ % 200 == 0:
        #     print(f"Iteration {_}/{max_iter}")

        for batch_y, batch_x in batch_iter(y_train, tx_train, batch_size, shuffle=True):
            gradient = logistic_loss_gradient(batch_y, batch_x, w, lambda_)
            w = w - gamma * gradient

        if all_losses:
            train_losses.append(logistic_loss(y_train, tx_train, w, lambda_))
            valid_losses.append(logistic_loss(y_valid, tx_valid, w, lambda_))

    if not all_losses:
        train_losses = logistic_loss(y_train, tx_train, w, lambda_)
        valid_losses = logistic_loss(y_valid, tx_valid, w, lambda_)

    return w, train_losses, valid_losses


def reg_logistic_regression_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    lambdas: np.ndarray,
    gammas: np.ndarray,
    w_initial: np.ndarray,
    max_iter: int = 5000,
    batch_size: int = 1,
):
    """
    GridSearchCV for regularized logistic regression.
    Return the optimum w and corresponding train loss, validation loss, lambda, gamma.

    Args:
        x_train:  shape=(N, D)
        y_train:  shape=(N, 1)
        x_valid:  shape=(N, D)
        y_valid:  shape=(N, 1)
        lambdas: an array of lambdas
        gammas: an array of gammas
        w_initial: an array of initial values for w
        max_iter: scalar
        batch_size: scalar

    Returns:
        best_w: numpy array, optimal weights.
        best_train_losses: scalar, train loss corresponding to the best_w
        best_valid_losses: scalar, validation loss corresponding to the best_w
        best_lambda: scalar, best value for lambda
        best_gamma: scalar, best value for gamma
    """
    train_losses_all = []
    valid_losses_all = []
    w_all = []

    for lambda_ in lambdas:
        for gamma in gammas:
            # Perform logistic regression with the current hyperparameters
            w, train_losses, valid_losses = reg_logistic_regression(
                tx_train=x_train,
                y_train=y_train,
                tx_valid=x_valid,
                y_valid=y_valid,
                lambda_=lambda_,
                max_iter=max_iter,
                gamma=gamma,
                batch_size=batch_size,
                w=w_initial,
            )
            w_all.append(w)
            train_losses_all.append(train_losses)
            valid_losses_all.append(valid_losses)

    # Find the index of the minimum valid loss
    best_loss_index = np.argmin(valid_losses_all)
    # Get the corresponding indices for lambda and gamma
    best_lambda_index, best_gamma_index = np.unravel_index(
        best_loss_index, (len(lambdas), len(gammas))
    )

    # Retrieve the corresponding hyperparameters
    best_lambda = lambdas[best_lambda_index]
    best_gamma = gammas[best_gamma_index]

    # Get the best weights and losses
    best_w = w_all[best_loss_index]
    best_train_losses = train_losses_all[best_loss_index]
    best_valid_losses = valid_losses_all[best_loss_index]

    return best_w, best_train_losses, best_valid_losses, best_lambda, best_gamma
