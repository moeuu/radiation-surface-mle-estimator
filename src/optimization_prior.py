import numpy as np
from calculate import *
from plot import *

def adjust_q_based_on_measurements(m_p, rad_all_measurements, shield_orientations, q_surfaces, x, y, z, min_q_value=1e-3):
    """
    測定値に基づいて適応的に q を調整し，prior 分布を適応的に修正
    """
    num_orientations = len(shield_orientations)
    updated_q = [q.copy().astype(np.float64) for q in q_surfaces]

    for idx, detector_position in enumerate(m_p):
        measurements = rad_all_measurements[idx * num_orientations:(idx + 1) * num_orientations]
        measurement_counts = {val: measurements.count(val) for val in set(measurements)}

        # 4回以上同じ測定値が出た場合、その方向の遮蔽領域を特定
        for measurement, count in measurement_counts.items():
            if count >= 4:
                for orientation_idx, orientation in enumerate(shield_orientations):
                    if measurements[orientation_idx] == measurement:
                        for surface_idx, q_surface in enumerate(updated_q):  
                            for grid_idx in range(len(q_surface)):
                                grid_position = get_grid_position(surface_idx, grid_idx, x, y, z)

                                # **デバッグ: 遮蔽判定の確認**
                                if shield_blocks_radiation(orientation, grid_position, detector_position):
                                    print(f"Updating q: Surface {surface_idx}, Grid {grid_idx}, Before: {updated_q[surface_idx][grid_idx]}")
                                    
                                    # **減衰率を大きく**
                                    updated_q[surface_idx][grid_idx] *= 0.99
                                    updated_q[surface_idx][grid_idx] = max(updated_q[surface_idx][grid_idx], min_q_value)
                                    
                                    print(f"Updated q: Surface {surface_idx}, Grid {grid_idx}, After: {updated_q[surface_idx][grid_idx]}")

    # **正規化をスキップ**
    # total_q = sum(q_surface.sum() for q_surface in updated_q)
    # if total_q > 1e-3:
    #     updated_q = [q_surface / total_q for q_surface in updated_q]

    return updated_q


def score_func_prior(A, b, q, mu, Sigma_inv, lam, alpha):
    """
    MAP 推定の目的関数 (Elastic Net 正則化付き)
    
    Parameters:
      A         : 計測行列
      b         : 観測データ
      q         : 推定パラメータ
      mu        : 事前平均
      Sigma_inv : 事前分布の共分散行列の逆行列
      lam       : 正則化の強さ（λ）
      alpha     : L1 と L2 のバランス（α）
                  α = 1 なら LASSO, α = 0 なら Ridge
                  
    Returns:
      score     : MAP 推定のスコア（大きいほど尤もらしい）
    """
    # Poisson 尤度項
    b_ave = A.dot(q)
    log_likelihood = np.sum(b * np.log(b_ave)) - np.sum(b_ave)

    
    # Elastic Net 事前分布項
    log_prior_L2 = -0.5 * lam * (1 - alpha) * np.dot((q - mu).T, np.dot(Sigma_inv, (q - mu)))
    log_prior_L1 = - lam * alpha * np.sum(np.abs(q))
    log_prior = log_prior_L2 + log_prior_L1
    print(log_prior_L2)
    print(log_prior_L1)
    
    score = log_likelihood + log_prior
    return score.item()

def grad_func_prior(A, b, q, mu, Sigma_inv, lam, alpha):
    """
    MAP 推定の勾配計算 (Elastic Net 正則化付き)
    
    Parameters:
      A, b, q, mu, Sigma_inv, lam, alpha : 各種パラメータ（score_func_prior の説明参照）
      
    Returns:
      grad : 目的関数の q に関する勾配
    """
    b_ave = A.dot(q)
    # Poisson 尤度の勾配: A^T (b / (Aq) - 1)
    grad_likelihood = A.T.dot((b.flatten() / b_ave.flatten()) - 1).reshape(q.shape)
    
    # L2 正則化項の勾配
    grad_L2 = - lam * (1 - alpha) * np.dot(Sigma_inv, (q - mu))
    
    # L1 正則化項のサブグラディエント
    grad_L1 = - lam * alpha * np.sign(q)
    
    return grad_likelihood + grad_L2 + grad_L1

def Adam_prior(A, b, q, mu, Sigma_inv, lam, alpha, x, y, z, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100000, sources=None):
    """
    Adam 最適化による MAP 推定（Elastic Net 正則化付き）
    """
    m = np.zeros_like(q)
    v = np.zeros_like(q)

    # `q` の各面の形状を定義
    q_shapes = [
        (x * y, 1),  # 地面 z=0
        (x * y, 1),  # 天井 z=10
        (y * z, 1),  # 側面1 x=0
        (x * z, 1),  # 側面2 y=0
        (y * z, 1),  # 側面3 x=10
        (x * z, 1)   # 側面4 y=10
    ]

    for i in range(max_iter):
        gradient = grad_func_prior(A, b, q, mu, Sigma_inv, lam, alpha)

        # Adam の更新
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)

        # バイアス補正
        m_hat = m / (1 - beta1**(i + 1))
        v_hat = v / (1 - beta2**(i + 1))

        # パラメータ更新
        q += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # q の下限制約（非負にする）
        q[q < 0] = 1e-6

        # 1000回ごとに 3D ヒートマップをプロット
        if i <=2000:
          if i % 1000 == 0:
              print(f"Iteration {i}, q_max: {q.max():.3f}, q_min: {q.min():.3f}")

              # `q` を `restore_q` を使って 6 面のリストに変換
              restored_q = restore_q(q, q_shapes)

              # カラーマップの範囲を統一
              all_q_values = np.concatenate([q_.flatten() for q_ in restored_q])
              vmin, vmax = all_q_values.min(), all_q_values.max()

              # 放射線源のリストを変換
              source_lists = [radiation_source(*src) for src in sources] if sources else None

              # 3D ヒートマップを描画
              plot_3d_heatmap(restored_q, x, y, z, sources=source_lists, vmin=0, vmax=vmax)

    return q


