import numpy as np
import random

def decide_measurement_points(x,y,r):
    m_p = []
    #z軸は固定
    m_p_c = []
    for i in range(x):
        for j in range(y):
            m_p_c.append([0.5+i,0.5+j,0.5])
    
    num_points = int(r * len(m_p_c))
    m_p = random.sample(m_p_c, num_points)
    return m_p

def shield_blocks_radiation(shield_orientation, source_position, detector_position):
    """
    改善された遮蔽判定関数
    """
    # 放射線源から検出器への方向ベクトル
    vector = np.array(source_position) - np.array(detector_position)
    
    # 球座標に変換
    theta, phi = cartesian_to_spherical(vector)
    
    # シールドの角度範囲
    shield_theta_min, shield_theta_max = shield_orientation['theta']
    shield_phi_min, shield_phi_max = shield_orientation['phi']
    
    # 角度がシールド内に入っているか確認
    in_theta = shield_theta_min <= theta <= shield_theta_max
    in_phi = shield_phi_min <= phi <= shield_phi_max
    
    return in_theta and in_phi

def cartesian_to_spherical(vector):
    """
    改善されたデカルト座標から球座標への変換
    Returns: theta (傾斜角), phi (方位角)
    """
    x, y, z = vector
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # theta（傾斜角）の計算
    theta = np.arccos(z / r) if r != 0 else 0
    
    # phi（方位角）の計算とその正規化
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    
    return theta, phi

# 減衰計算
def calculate_attenuation(shield_orientation, source_position, detector_position):
    """
    放射線源から検出器までの減衰を計算する関数。
    """
    # 距離による基本的な減衰（逆二乗則）
    distance = np.linalg.norm(np.array(source_position) - np.array(detector_position))
    base_attenuation = 1 / (distance ** 2)

    shield_attenuation = 0.10177304964539008

    # シールドが放射線経路を遮るかチェック
    if shield_blocks_radiation(shield_orientation, source_position, detector_position):
        attenuation = base_attenuation * shield_attenuation  # 例: 80%の減衰が適用される
    else:
        attenuation = base_attenuation
    
    return attenuation

def calculate_distance(x1, y1, z1, x2, y2, z2):
    """2点間の距離を計算"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def measurement_shield(m_p, source_list, shield_orientations):
    """
    各計測点ごとに遮蔽体を8回の向きで計測し、合計放射線量を求める関数。
    m_p: 計測点リスト [(x, y, z), ...]（検出器の位置）
    source_list: 放射線源リスト [(x, y, z, intensity), ...]
    shield_orientations: 8つの遮蔽体の向きリスト
    """
    rad_all_measurements = []

    for detector_position in m_p:
        for orientation in shield_orientations:
            temp_rad = 0
            for source in source_list:
                # sourceの位置と強度を取得
                source_position = source[:3]
                intensity = source[3]

                # 距離と遮蔽体による減衰を計算
                distance = calculate_distance(detector_position[0], detector_position[1], detector_position[2],
                                              source_position[0], source_position[1], source_position[2])
                attenuation = calculate_attenuation(orientation, source_position, detector_position)

                # 減衰を含めた放射線量を加算
                temp_rad += intensity * attenuation

            rad_all_measurements.append(temp_rad)

    return rad_all_measurements

# グリッド生成関数
def create_grid(axis, position, g=1):
    # 部屋のサイズとグリッドサイズ
    size_x = 10
    size_y = 10
    size_z = 10
    """
    指定された面を 10x10 のグリッドで分割し、各グリッドの中心点をリストとして返す。
    axis: グリッドを生成する軸 ('x', 'y', 'z')
    position: 軸上の位置 (0, 10など)
    """
    G = []
    if axis == 'x':  # 側面 x=position
        for i in range(size_y * size_z):
            G_y = (i % size_y) * g + g / 2 
            G_z = (i // size_y) * g + g / 2
            G.append([position, G_y, G_z])
    elif axis == 'y':  # 側面 y=position
        for i in range(size_x * size_z):
            G_x = (i % size_x) * g + g / 2 
            G_z = (i // size_x) * g + g / 2
            G.append([G_x, position, G_z])
    elif axis == 'z':  # 側面 z=position
        for i in range(size_x * size_y):
            G_x = (i % size_x) * g + g / 2 
            G_y = (i // size_x) * g + g / 2
            G.append([G_x, G_y, position])

    return G

def create_A(l1, l2, m_p, G, shield_orientations):
    """
    A行列を生成する関数。
    l1, l2: グリッドの辺の長さ
    m_p: 計測点のリスト [(x, y, z)]
    G: グリッド点のリスト [(x, y, z), ...]
    shield_orientations: シールドの向きリスト
    """
    g = 1
    d1 = l1 // g  # グリッドの個数 (整数除算)
    d2 = l2 // g
    n_m = len(m_p)  # 計測点の数
    A = np.zeros((n_m * len(shield_orientations), int(d1 * d2)))  # A行列の初期化 (800, 100)

    # 各計測点とシールドの回転ごとにループ
    for i in range(n_m):
        m_x, m_y, m_z = m_p[i]

        for k, orientation in enumerate(shield_orientations):
            row_index = i * len(shield_orientations) + k  # 各回転動作での行インデックス

            # グリッドごとにループ
            for j in range(int(d1 * d2)):
                G_x, G_y, G_z = G[j]
                distance = calculate_distance(m_x, m_y, m_z, G_x, G_y, G_z)

                # 逆二乗則に基づく基本的な減衰
                A[row_index, j] = 1 / (distance ** 2) if distance != 0 else 0

                # シールドによる減衰を適用
                if shield_blocks_radiation(orientation, [G_x, G_y, G_z], [m_x, m_y, m_z]):
                    attenuation = calculate_attenuation(orientation, [G_x, G_y, G_z], [m_x, m_y, m_z])
                    A[row_index, j] *= attenuation

    return A


    # 線源に最も近いグリッドとそのインデックスを計算
def find_nearest_grid(source, grid):
    min_distance = float('inf')
    nearest_grid_index = -1
    nearest_grid_point = None
    
    for index, point in enumerate(grid):
        distance = calculate_distance(source[0], source[1], source[2], point[0], point[1], point[2])
        if distance < min_distance:
            min_distance = distance
            nearest_grid_index = index
            nearest_grid_point = point
            
    return nearest_grid_point, nearest_grid_index

def restore_q(q_optimized, q_shapes):
    restored_q = []
    index = 0
    for shape in q_shapes:
        length = np.prod(shape)  # 各qの要素数
        restored_q.append(q_optimized[index:index + length].reshape(shape))
        index += length
    return restored_q

def get_grid_position(surface_idx, grid_idx, x, y, z, g=1):
    """
    各面のインデックスとグリッドインデックスからグリッドの中心位置 (x, y, z) を取得する関数。
    surface_idx: 面のインデックス（0-5）
    grid_idx: グリッドのインデックス
    x, y, z: 部屋の寸法
    g: グリッドサイズ（デフォルトは1）
    """
    if surface_idx == 0:  # 地面 z=0
        gx = (grid_idx % x) * g + g / 2 + 0.000001
        gy = (grid_idx // x) * g + g / 2 + 0.000001
        gz = 0
    elif surface_idx == 1:  # 天井 z=z
        gx = (grid_idx % x) * g + g / 2 + 0.000001
        gy = (grid_idx // x) * g + g / 2 + 0.000001
        gz = z
    elif surface_idx == 2:  # 側面1 x=0
        gx = 0
        gy = (grid_idx % y) * g + g / 2 + 0.000001
        gz = (grid_idx // y) * g + g / 2 + 0.000001
    elif surface_idx == 3:  # 側面2 y=0
        gx = (grid_idx % x) * g + g / 2 + 0.000001
        gy = 0
        gz = (grid_idx // x) * g + g / 2 + 0.000001
    elif surface_idx == 4:  # 側面3 x=x
        gx = x
        gy = (grid_idx % y) * g + g / 2 + 0.000001
        gz = (grid_idx // y) * g + g / 2 + 0.000001
    elif surface_idx == 5:  # 側面4 y=y
        gx = (grid_idx % x) * g + g / 2 + 0.000001
        gy = y
        gz = (grid_idx // x) * g + g / 2 + 0.000001
    return np.array([gx, gy, gz])