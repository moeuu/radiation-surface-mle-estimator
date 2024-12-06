import numpy as np
def score_func(A, b, q):
    b_ave = A.dot(q)  # 推定bの平均
    score = np.sum(b * np.log(b_ave)) - np.sum(b_ave)
    return score

# gradient function

def grad_func(A, b, q):
    
    b_ave = A.dot(q)  # 推定bの平均
    grad_tmp = (b.flatten() / b_ave.flatten())[:, np.newaxis] * A
    grad = grad_tmp.sum(axis=0) - A.sum(axis=0)
    return grad.reshape(q.shape)

def Adam(A, b, q, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
    m = np.zeros_like(q)
    v = np.zeros_like(q)
    
    for i in range(max_iter):
        t = i + 1
        gradient = grad_func(A, b, q)
        
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
        
        # スコアの更新
        score = score_func(A, b, q)
    
    return q