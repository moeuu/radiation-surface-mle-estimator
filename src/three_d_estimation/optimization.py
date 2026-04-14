import numpy as np

from .config import OptimizerConfig


def _clip_prediction(prediction, epsilon=1e-12):
    return np.clip(prediction, epsilon, None)


def score_func(A, b, q):
    b_ave = _clip_prediction(A.dot(q))
    return float(np.sum(b * np.log(b_ave)) - np.sum(b_ave))


def grad_func(A, b, q):
    b_ave = _clip_prediction(A.dot(q))
    return A.T.dot((b.flatten() / b_ave.flatten()) - 1).reshape(q.shape)


def adam_optimize(A, b, q, optimizer_config):
    m = np.zeros_like(q)
    v = np.zeros_like(q)

    for iteration in range(optimizer_config.max_iter):
        gradient = grad_func(A, b, q)
        m = optimizer_config.beta1 * m + (1 - optimizer_config.beta1) * gradient
        v = optimizer_config.beta2 * v + (1 - optimizer_config.beta2) * (gradient**2)

        m_hat = m / (1 - optimizer_config.beta1 ** (iteration + 1))
        v_hat = v / (1 - optimizer_config.beta2 ** (iteration + 1))

        q += optimizer_config.learning_rate * m_hat / (np.sqrt(v_hat) + optimizer_config.epsilon)
        q[q < optimizer_config.min_q_value] = optimizer_config.min_q_value

    return q


def Adam(
    A,
    b,
    q,
    x=None,
    y=None,
    z=None,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    max_iter=100000,
    sources=None,
    optimizer_config=None,
):
    del x, y, z, sources
    if optimizer_config is None:
        optimizer_config = OptimizerConfig(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_iter=max_iter,
        )
    return adam_optimize(A, b, q, optimizer_config)
