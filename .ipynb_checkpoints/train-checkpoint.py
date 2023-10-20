import numpy as np

def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, loss: str = 'MSE'):
    """Calculate the loss using MSE or MAE.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
        loss: either "MSE" or "MAE"

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    assert loss in ["MSE", "MAE"]
    err = y - tx.dot(w)
    N = y.shape[0]

    if loss == "MSE":
        return 0.5 * np.power(err, 2).sum()/N
    elif loss == "MAE":
        return np.sum(np.abs(y - np.matmul(tx, w)))/N


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray, loss: str):
    """Computes the gradient at w (for the MSE/MAE).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        loss: either "MAE" or "MSE"

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    assert loss in ["MAE", "MSE"]
    N = y.shape[0]
    err = y - np.matmul(tx, w)

    if loss == "MSE":
        grad = -(np.matmul(tx.transpose(), err))/N
    else:
        raise NotImplementedError

    return grad


def gradient_descent(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float,
                     loss_type: str ="MSE"):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        loss: either "MSE" or "MAE"

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    assert loss_type in ["MSE", "MAE"]
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        grad = compute_gradient(y, tx, w, loss_type)
        loss_val = compute_loss(y, tx, w, loss_type)

        w = w - gamma*grad
        # ***************************************************

        # store w and loss
        ws.append(w)
        losses.append(loss_val)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss_val, w0=w[0], w1=w[1]
            )
        )
    return losses, ws

def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for yb, xb in batch_iter(y,tx):
            # compute gradient on batch and loss
            stochastic_grad=compute_gradient(yb,xb,w)
            loss=compute_loss(yb,xb,w)
            # update w
            w=w-gamma*stochastic_grad

            ws.append(w)
            losses.append(loss)

    return losses, ws


def least_squares(y, tx):
    w_star = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    loss = compute_loss(y,tx,w_star)
    return w_star, loss

def ridge_regression(y, tx, _lambda):
    N = int(tx.size/tx.shape[0])
    lambda__ = lambda_*2*N
    return np.linalg.solve(tx.T.dot(tx)+lambda__*np.eye(N), tx.T.dot(y))