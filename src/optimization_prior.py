import numpy as np
from calculate import *
import numpy as np

import numpy as np

import numpy as np

import numpy as np

def compute_prior_distribution_sparse(updated_q, x=10, y=10, z=10, grid_size = 1.0,
                                     lambda_corr=0.7, lambda_shield=2.0, 
                                     epsilon_factor=3e-3, sparse_threshold=0.004):
    """
    スパース性を考慮したガウス事前分布モデル（座標系を用いた隣接性判定 + 調整可能なハイパーパラメータ）
    
    - updated_q: 遮蔽回転測定によって更新された q（リスト形式、6面分）
    - x, y, z: 部屋の寸法（グリッド数）
    - lambda_corr: 隣接グリッドとの相関強度（通常領域）
    - lambda_shield: 遮蔽の影響を受けたグリッドの不確実性を増加させる係数
    - epsilon_factor: 数値安定性のための正則化係数（デフォルト 1e-3）
    - sparse_threshold: これ以上の q の値を持つグリッドを "点線源の可能性が高い" とみなす閾値

    Returns:
    - mu: 事前分布の平均ベクトル
    - Sigma_inv: 逆共分散行列（スパース性を考慮）
    """
    # 平均ベクトル μ を作成
    q_flatten = np.concatenate([q.flatten() for q in updated_q])
    mu = q_flatten.copy()

    # 共分散行列 Σ の作成
    sigma_q = np.std(q_flatten)
    epsilon = max(1e-4, sigma_q * epsilon_factor)
    grid_count = len(q_flatten)

    # 初期化
    Sigma = np.eye(grid_count) * epsilon  

    # インデックスマップの作成（6面のグリッドを1Dに展開）
    index_map = {}
    grid_positions = {}  # 座標マップ
    idx = 0

    for surface_idx, q_surface in enumerate(updated_q):
        for grid_idx in range(len(q_surface)):
            index_map[(surface_idx, grid_idx)] = idx
            grid_positions[(surface_idx, grid_idx)] = get_grid_position(surface_idx, grid_idx, x, y, z)
            idx += 1

    # スパース性を考慮した共分散行列の調整
    for (s1, g1), idx1 in index_map.items():
        q_val1 = updated_q[s1][g1]  # 現在の q の値
        pos1 = np.array(grid_positions[(s1, g1)])

        for (s2, g2), idx2 in index_map.items():
            q_val2 = updated_q[s2][g2]
            pos2 = np.array(grid_positions[(s2, g2)])

            # **座標ベースで隣接性を判定**
            if np.linalg.norm(pos1 - pos2) <= 1.01 * grid_size:
                if q_val1 > sparse_threshold or q_val2 > sparse_threshold:
                    # 点線源の可能性が高い場合は相関を減らす（スパース性を考慮）
                    Sigma[idx1, idx2] += lambda_corr * 0.1
                else:
                    # 通常のグリッド
                    Sigma[idx1, idx2] += lambda_corr

        # **自己相関の調整**
        if q_val1 > sparse_threshold:
            Sigma[idx1, idx1] += lambda_shield * 10  # 高い自己相関（スパース性を反映）
        else:
            Sigma[idx1, idx1] += lambda_shield

    # 数値安定性のための正則化
    Sigma += np.eye(grid_count) * epsilon

    # 逆共分散行列の計算（擬似逆行列）
    Sigma_inv = np.linalg.pinv(Sigma)

    return mu, Sigma_inv


def score_func(A, b, q, mu, Sigma_inv):
    """
    放射線源の MAP 推定の目的関数
    - Poisson尤度 + 事前分布
    """
    b_ave = A.dot(q)  
    log_likelihood = np.sum(b * np.log(b_ave)) - np.sum(b_ave)  # Poisson log-likelihood

    # 事前分布の対数 (正規分布)
    lambda_prior = 0.0001  # Prior の影響を調整
    log_prior = -0.5 * lambda_prior * np.dot((q - mu).T, np.dot(Sigma_inv, (q - mu)))  
    print(log_prior)
    
    # MAP 推定のスコア
    score = log_likelihood + log_prior
    return score.item()

def grad_func(A, b, q, mu, Sigma_inv):
    """
    MAP 推定の勾配計算 (Poisson 尤度 + 事前分布)
    """
    b_ave = A.dot(q)  
    grad_tmp = (b.flatten() / b_ave.flatten())[:, np.newaxis] * A
    grad_likelihood = grad_tmp.sum(axis=0).reshape(q.shape) - A.sum(axis=0).reshape(q.shape)  # 修正点: `.reshape(q.shape)`

    lambda_prior = 0.0001  # Prior の影響を調整
    grad_prior = -lambda_prior * np.dot(Sigma_inv, (q - mu)).reshape(q.shape)  # `@` → `np.dot()`

    return (grad_likelihood + grad_prior)

def Adam(A, b, q, mu, Sigma_inv, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100000):
    """
    Adam 最適化 (MAP 推定)
    """
    m = np.zeros_like(q)
    v = np.zeros_like(q)

    for i in range(max_iter):
        gradient = grad_func(A, b, q, mu, Sigma_inv)  

        # モーメント更新
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)

        # バイアス補正
        m_hat = m / (1 - beta1**(i + 1))
        v_hat = v / (1 - beta2**(i + 1))

        # パラメータ更新
        q += learning_rate * m_hat / (np.sqrt(v_hat + epsilon))

        # q の下限制約
        q[q < 0] = 1e-6

    return q
