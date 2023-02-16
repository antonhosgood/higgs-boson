"""Module containing implementations of loss functions and machine learning methods."""

import numpy as np


def calculate_loss_mse(y, tx, w):
    """Computes the loss using MSE.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        w: shape=(D,1). The vector of model parameters.

    Returns:
        The value of the loss (a scalar), corresponding to the input parameters `w`.
    """

    # Convert from 1D to 2D array
    if len(tx.shape) == 1:
        tx = tx[np.newaxis]

    err = y - tx.dot(w)
    return 0.5 * np.mean(err**2)


def calculate_gradient_mse(y, tx, w):
    """Computes the gradient at `w` using MSE.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        w: shape=(D,1). The vector of model parameters.

    Returns:
        An array of shape (D, ), containing the gradient of the loss at `w`.
    """

    # Convert from 1D to 2D array
    if len(tx.shape) == 1:
        tx = tx[np.newaxis]

    err = y - tx.dot(w)
    return -tx.T.dot(err) / len(err)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: shape=(D,1). Initial guess (or the initialisation) of the model parameters.
        max_iters: Scalar denoting the total number of iterations of GD.
        gamma: Scalar denoting the stepsize.

    Returns:
        w: Weight vector from the final iteration of GD.
        loss: Corresponding loss value of `w`.
    """

    w = initial_w
    for _ in range(max_iters):
        grad = calculate_gradient_mse(y, tx, w)
        w = w - gamma * grad

    loss = calculate_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Linear regression using stochastic gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: shape=(D,1). Initial guess (or the initialisation) of the model parameters.
        max_iters: Scalar denoting the total number of iterations of SGD.
        gamma: Scalar denoting the stepsize.
        batch_size: Scalar denoting the mini-batch size.

    Returns:
        w: Weight vector from the final iteration of SGD.
        loss: Corresponding loss value of `w`.
    """

    w = initial_w
    for _ in range(max_iters):
        idxs = np.random.permutation(len(y))[:batch_size]
        grad = calculate_gradient_mse(y[idxs], tx[idxs], w)
        w = w - gamma * grad

    loss = calculate_loss_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: numpy array of shape (N,1). N is the number of samples.
        tx: numpy array of shape (N,D). D is the number of features.

    Returns:
        w: Optimal weights.
        loss: Corresponding loss value of `w`.
    """

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = calculate_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: numpy array of shape (N,1). N is the number of samples.
        tx: numpy array of shape (N,D). D is the number of features.
        lambda_: Scalar value of regularisation factor.

    Returns:
        w: Optimal weights.
        loss: Corresponding loss value of `w`.
    """

    a = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = calculate_loss_mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """Apply sigmoid function on t."""

    return 1 / (1 + np.exp(-t))


def calculate_loss_log(y, tx, w):
    """Computes the loss using log loss.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        w: shape=(D,1). The vector of model parameters.

    Returns:
        The value of the loss (a scalar), corresponding to the input parameters `w`.
    """

    # Convert from 1D to 2D array
    if len(tx.shape) == 1:
        tx = tx[np.newaxis]

    pred = sigmoid(tx.dot(w))
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))


def calculate_gradient_log(y, tx, w):
    """Computes the gradient at `w` using log loss.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        w: shape=(D,1). The vector of model parameters.

    Returns:
        An array of shape (D,1), containing the gradient of the loss at `w`.
    """

    # Convert from 1D to 2D array
    if len(tx.shape) == 1:
        tx = tx[np.newaxis]

    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: shape=(D,1). Initial guess (or the initialisation) of the model parameters.
        max_iters: Scalar denoting the total number of iterations of GD.
        gamma: Scalar denoting the stepsize.

    Returns:
        w: Weight vector from the final iteration of GD.
        loss: Corresponding loss value of `w`.
    """

    w = initial_w
    for _ in range(max_iters):
        grad = calculate_gradient_log(y, tx, w)
        w = w - gamma * grad

    loss = calculate_loss_log(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularised logistic regression using gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        lambda_: Scalar value of regularisation factor.
        initial_w: shape=(D,1). Initial guess (or the initialisation) of the model parameters.
        max_iters: Scalar denoting the total number of iterations of GD.
        gamma: Scalar denoting the stepsize.

    Returns:
        w: Weight vector from the final iteration of GD.
        loss: Corresponding loss value of `w`.
    """

    w = initial_w
    for _ in range(max_iters):
        grad = calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    loss = calculate_loss_log(y, tx, w)
    return w, loss


def logistic_regression_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Logistic regression using stochastic gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: shape=(D,1). Initial guess (or the initialisation) of the model parameters.
        max_iters: Scalar denoting the total number of iterations of GD.
        gamma: Scalar denoting the stepsize.
        batch_size: Scalar denoting the mini-batch size.

    Returns:
        w: Weight vector from the final iteration of SGD.
        loss: Corresponding loss value of `w`.
    """

    w = initial_w
    for _ in range(max_iters):
        idxs = np.random.permutation(len(y))[:batch_size]
        grad = calculate_gradient_log(y[idxs], tx[idxs], w)
        w = w - gamma * grad

    loss = calculate_loss_log(y, tx, w)
    return w, loss


def lasso_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularised logistic regression using gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        lambda_: Scalar value of regularisation factor.
        initial_w: shape=(D,1). Initial guess (or the initialisation) of the model parameters.
        max_iters: Scalar denoting the total number of iterations of GD.
        gamma: Scalar denoting the stepsize,

    Returns:
        w: Weight vector from the final iteration of GD.
        loss: Corresponding loss value of `w`.
    """

    w = initial_w
    for _ in range(max_iters):
        grad = calculate_gradient_log(y, tx, w) + lambda_ * np.sign(w)
        w = w - gamma * grad

    loss = calculate_loss_log(y, tx, w)
    return w, loss
