import numpy as np

from .config import OptimizerConfig
from .geometry import get_grid_position
from .measurement import shield_blocks_radiation


def adjust_q_based_on_measurements(
    measurement_points,
    measurements_all,
    shield_orientations,
    q_surfaces,
    x,
    y,
    z,
    min_q_value=1e-3,
):
    num_orientations = len(shield_orientations)
    updated_q = [surface.copy().astype(np.float64) for surface in q_surfaces]

    for measurement_index, detector_position in enumerate(measurement_points):
        start = measurement_index * num_orientations
        stop = (measurement_index + 1) * num_orientations
        measurements = list(measurements_all[start:stop])
        measurement_counts = {value: measurements.count(value) for value in set(measurements)}

        for measurement_value, count in measurement_counts.items():
            if count < 4:
                continue
            for orientation_index, orientation in enumerate(shield_orientations):
                if measurements[orientation_index] != measurement_value:
                    continue
                for surface_index, q_surface in enumerate(updated_q):
                    for grid_index in range(len(q_surface)):
                        grid_position = get_grid_position(surface_index, grid_index, x, y, z)
                        if shield_blocks_radiation(orientation, grid_position, detector_position):
                            q_surface[grid_index] *= 0.99
                            q_surface[grid_index] = max(q_surface[grid_index], min_q_value)

    return updated_q


def compute_prior_distribution_sparse(
    updated_q,
    lambda_corr=0.1,
    lambda_shield=10.0,
    grid_size=1.0,
    sparse_threshold=0.004,
):
    del grid_size

    q_flatten = np.concatenate([q.flatten() for q in updated_q])
    mu = q_flatten.copy()
    sigma_q = np.std(q_flatten)
    epsilon = max(1e-4, sigma_q * 0.01)
    grid_count = len(q_flatten)
    sigma = np.eye(grid_count) * epsilon

    index_map = {}
    current_index = 0
    for surface_idx, q_surface in enumerate(updated_q):
        for grid_idx in range(len(q_surface)):
            index_map[(surface_idx, grid_idx)] = current_index
            current_index += 1

    for (surface_idx_1, grid_idx_1), idx_1 in index_map.items():
        q_val_1 = updated_q[surface_idx_1][grid_idx_1]
        for (surface_idx_2, grid_idx_2), idx_2 in index_map.items():
            q_val_2 = updated_q[surface_idx_2][grid_idx_2]

            if surface_idx_1 == surface_idx_2 and abs(grid_idx_1 - grid_idx_2) == 1:
                sigma[idx_1, idx_2] += lambda_corr * 0.1 if (
                    q_val_1 > sparse_threshold or q_val_2 > sparse_threshold
                ) else lambda_corr

            if idx_1 == idx_2:
                sigma[idx_1, idx_2] += lambda_shield * 10 if q_val_1 > sparse_threshold else lambda_shield

    sigma += np.eye(grid_count) * epsilon
    sigma_inv = np.linalg.pinv(sigma)
    return mu, sigma_inv


def score_func_prior(A, b, q, mu, Sigma_inv, lam, alpha):
    b_ave = np.clip(A.dot(q), 1e-12, None)
    log_likelihood = np.sum(b * np.log(b_ave)) - np.sum(b_ave)
    log_prior_l2 = -0.5 * lam * (1 - alpha) * np.dot((q - mu).T, np.dot(Sigma_inv, (q - mu)))
    log_prior_l1 = -lam * alpha * np.sum(np.abs(q))
    return float((log_likelihood + log_prior_l2 + log_prior_l1).item())


def grad_func_prior(A, b, q, mu, Sigma_inv, lam, alpha):
    b_ave = np.clip(A.dot(q), 1e-12, None)
    grad_likelihood = A.T.dot((b.flatten() / b_ave.flatten()) - 1).reshape(q.shape)
    grad_l2 = -lam * (1 - alpha) * np.dot(Sigma_inv, (q - mu))
    grad_l1 = -lam * alpha * np.sign(q)
    return grad_likelihood + grad_l2 + grad_l1


def Adam_prior(
    A,
    b,
    q,
    mu,
    Sigma_inv,
    lam,
    alpha,
    x,
    y,
    z,
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
            min_q_value=1e-6,
        )

    m = np.zeros_like(q)
    v = np.zeros_like(q)

    for iteration in range(optimizer_config.max_iter):
        gradient = grad_func_prior(A, b, q, mu, Sigma_inv, lam, alpha)
        m = optimizer_config.beta1 * m + (1 - optimizer_config.beta1) * gradient
        v = optimizer_config.beta2 * v + (1 - optimizer_config.beta2) * (gradient**2)

        m_hat = m / (1 - optimizer_config.beta1 ** (iteration + 1))
        v_hat = v / (1 - optimizer_config.beta2 ** (iteration + 1))

        q += optimizer_config.learning_rate * m_hat / (np.sqrt(v_hat) + optimizer_config.epsilon)
        q[q < optimizer_config.min_q_value] = optimizer_config.min_q_value

    return q
