import numpy as np
from calculate import *
from plot import *

def score_func(A, b, q):
    b_ave = A.dot(q)  # 推定bの平均
    score = np.sum(b * np.log(b_ave)) - np.sum(b_ave)
    return score

def grad_func(A, b, q):
    b_ave = A.dot(q)  # 推定bの平均
    grad = A.T.dot((b.flatten() / b_ave.flatten()) - 1).reshape(q.shape)
    return grad

def Adam(A, b, q, x, y, z, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100000, sources=None):
    """
    Adam 最適化による放射線分布の推定
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
        gradient = grad_func(A, b, q)

        # Adam の更新
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)

        # バイアス補正
        m_hat = m / (1 - beta1**(i + 1))
        v_hat = v / (1 - beta2**(i + 1))

        # パラメータ更新
        q += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # q の下限制約（非負にする）
        q[q < 0] = 1e-7

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

