import numpy as np

def score_func_prior(A, b, q, prior_q):
    """
    MAP推定用スコア関数: 尤度 + 事前分布の対数
    """
    b_ave = A.dot(q)  # 推定bの平均
    log_likelihood = np.sum(b * np.log(b_ave)) - np.sum(b_ave)
    
    # 事前分布（遮蔽体情報による更新済み事前分布）の対数
    log_prior = -np.sum((q - prior_q)**2)  # 二乗誤差として事前分布を考慮
    
    return log_likelihood + log_prior

def grad_func_prior(A, b, q, prior_q):
    """
    MAP推定用勾配関数: 尤度の勾配 + 事前分布の勾配
    """
    b_ave = A.dot(q)  # 推定bの平均
    grad_likelihood_tmp = (b.flatten() / b_ave.flatten())[:, np.newaxis] * A
    grad_likelihood = grad_likelihood_tmp.sum(axis=0) - A.sum(axis=0)
    
    # 事前分布（遮蔽体情報による更新済み事前分布）の勾配
    grad_prior = -2 * (q - prior_q)  # 二乗誤差の勾配
    
    return grad_likelihood.reshape(q.shape) + grad_prior

def Adam_prior(A, b, q, prior_q, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
    """
    Adam最適化アルゴリズムを用いたMAP推定
    """
    m = np.zeros_like(q)
    v = np.zeros_like(q)
    
    for i in range(max_iter):
        t = i + 1
        gradient = grad_func_prior(A, b, q, prior_q)
        
        # モーメントの更新
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)
        
        # バイアス補正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # パラメータの更新
        q += learning_rate * m_hat / (np.sqrt(v_hat + epsilon))
        
        # 下限制約: 0より大きく設定
        q[q < 0] = 0.0000001
        
        # スコアの更新（デバッグ用）
        score = score_func_prior(A, b, q, prior_q)
        if i % 100 == 0:
            print(f"Iteration {i}, Score: {score:.4f}")
    
    return q
