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

def batch_iter(y: np.ndarray, tx: np.ndarray, batch_size: int, num_batches: int = 1, shuffle: bool = True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    N = len(y)
    grad = -tx.T.dot(err) / N
    return grad 

def mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: int):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        # update w
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws

def mean_squared_error_sgd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: int):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for mini_batch_y, mini_batch_tx in batch_iter(y=y, tx=tx, batch_size=batch_size, shuffle=True):
            continue
            # compute gradient on batch and loss
            stochastic_grad = compute_gradient(mini_batch_y,mini_batch_tx,w)
            loss = compute_loss(mini_batch_y,mini_batch_tx,w)
            # update w
            w = w - gamma * stochastic_grad
            # store w and loss
            ws.append(w)
            losses.append(loss)

    return losses, ws

def least_squares(y: np.ndarray, tx: np.ndarray):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    loss = compute_loss(y,tx,w)
    return w, loss

def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: int):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    if tx.ndim == 1:
        N = tx.shape[0]
        A = np.array([[np.dot(tx.T, tx) + 2 * N * lambda_]])
        b = np.array([np.dot(tx.T, y)])
    else:
        N, D = tx.shape
        A = np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D)
        b = np.dot(tx.T, y)
    opt_w = np.linalg.solve(A, b)
    return opt_w