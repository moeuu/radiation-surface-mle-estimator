import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_measurement_points(m_points, sources, title, x, y):
    """
    Plot the measurement points.

    Parameters
    ----------
    m_points : list of lists
        List of measurement points.
    sources : list of lists
        List of sources.
    title : str
        Title of the plot.
    x : float
        Length of the room in the x direction.
    y : float
        Length of the room in the y direction.
    """
    x_ns, y_ns = zip(*[(p[0], p[1]) for p in m_points])
    plt.scatter(x_ns, y_ns, color='blue', label='Measurement points')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(0, x)
    plt.ylim(0, y)
    plt.grid(True, linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')  # x軸とy軸のスケールを等しく設定
    plt.show()

def plot_measurement_points_3d(m_points, title, x, y, z):
    # 仮のゼロ分布（部屋表示のためだけに使う）
    dummy_qs = [
        np.zeros((y, x)),  # bottom
        np.zeros((y, x)),  # top
        np.zeros((z, y)),  # x=0
        np.zeros((z, x)),  # y=0
        np.zeros((z, y)),  # x=max
        np.zeros((z, x))   # y=max
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=30, roll=0)

    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = "Greys"

    # --- XY 平面（地面・天井） ---
    Xb, Yb = np.meshgrid(np.linspace(0, x, x+1), np.linspace(0, y, y+1))
    ground_pad = np.pad(dummy_qs[0], ((0, 1), (0, 1)), mode='edge')
    ceiling_pad = np.pad(dummy_qs[1], ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(Xb, Yb, np.zeros_like(Xb), facecolors=get_transparent_colors(ground_pad, norm, cmap), shade=False)
    ax.plot_surface(Xb, Yb, np.full_like(Xb, z), facecolors=get_transparent_colors(ceiling_pad, norm, cmap), shade=False)

    # --- YZ 平面（x=0, x=x） ---
    Yb_side, Zb_side = np.meshgrid(np.linspace(0, y, y+1), np.linspace(0, z, z+1))
    side1_pad = np.pad(dummy_qs[2], ((0, 1), (0, 1)), mode='edge')
    side3_pad = np.pad(dummy_qs[4], ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(np.full_like(Yb_side, 0), Yb_side, Zb_side, facecolors=get_transparent_colors(side1_pad, norm, cmap), shade=False)
    ax.plot_surface(np.full_like(Yb_side, x), Yb_side, Zb_side, facecolors=get_transparent_colors(side3_pad, norm, cmap), shade=False)

    # --- XZ 平面（y=0, y=y） ---
    Xb_side, Zb_side = np.meshgrid(np.linspace(0, x, x+1), np.linspace(0, z, z+1))
    side2_pad = np.pad(dummy_qs[3], ((0, 1), (0, 1)), mode='edge')
    side4_pad = np.pad(dummy_qs[5], ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(Xb_side, np.full_like(Xb_side, 0), Zb_side, facecolors=get_transparent_colors(side2_pad, norm, cmap), shade=False)
    ax.plot_surface(Xb_side, np.full_like(Xb_side, y), Zb_side, facecolors=get_transparent_colors(side4_pad, norm, cmap), shade=False)

    # --- 計測点の描画 ---
    if m_points:
        x_ns, y_ns, z_ns = zip(*m_points)
        ax.scatter(x_ns, y_ns, z_ns, color='blue', label='Measurement Points', s=40, zorder=10)

    # --- 線源の描画（m_points とは別に source_* を読み取る）---
    try:
        global source_list  # 外部の source_list を利用
        for idx, source in enumerate(source_list):
            sx, sy, sz = source[:3]
            z_pos = sz if sz > 0 else 0.1
            ax.scatter(sx, sy, z_pos, color='red', s=100, edgecolors='black',
                       label='Radiation Source' if idx == 0 else None, zorder=20)
            ax.text(sx + 0.5, sy, z_pos + 0.8, f'{int(source[3])} [Bq]', color='black', fontsize=20, zorder=30)
    except NameError:
        pass  # source_list が未定義なら無視

    # --- 軸と見た目設定 ---
    ax.set_xlabel('X [m]', fontsize=15)
    ax.set_ylabel('Y [m]', fontsize=15)
    ax.set_zlabel('Z [m]', fontsize=15)
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)
    ax.set_xticks(np.arange(0, x+1, 2))
    ax.set_yticks(np.arange(0, y+1, 2))
    ax.set_zticks(np.arange(0, z+1, 2))
    ax.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.7)

        # --- 1mごとの点線グリッド（plot_3d_heatmapと同じ） ---
    step = 1
    lw = 0.5
    for i in np.linspace(0, x, int(x)+1):
        ax.plot([i, i], [0, y], [z, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for j in np.linspace(0, y, int(y)+1):
        ax.plot([0, x], [j, j], [z, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for k in np.linspace(0, z, int(z)+1):
        ax.plot([x, x], [0, y], [k, k], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for j in np.linspace(0, y, int(y)+1):
        ax.plot([x, x], [j, j], [0, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for k in np.linspace(0, z, int(z)+1):
        ax.plot([0, x], [y, y], [k, k], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for i in np.linspace(0, x, int(x)+1):
        ax.plot([i, i], [y, y], [0, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)


    # 立方体の枠線
    edges = [
        [[0, 0], [0, y], [0, 0]], [[0, x], [0, 0], [0, 0]], [[x, x], [0, y], [0, 0]], [[0, x], [y, y], [0, 0]],
        [[0, 0], [0, y], [z, z]], [[0, x], [0, 0], [z, z]], [[x, x], [0, y], [z, z]], [[0, x], [y, y], [z, z]],
        [[0, 0], [0, 0], [0, z]], [[x, x], [0, 0], [0, z]], [[0, 0], [y, y], [0, z]], [[x, x], [y, y], [0, z]],
    ]
    for edge in edges:
        ax.plot(edge[0], edge[1], edge[2], color="black", linewidth=1, zorder=1)

    plt.title(title)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_heatmap(q, shape, extent, title, sources=None, vmin=None, vmax=None):
    q_grid = q.reshape(shape)
    plt.figure()
    plt.imshow(q_grid, origin='lower', cmap='Greys', extent=extent, vmin=vmin, vmax=vmax)
    plt.colorbar(label='[Bq]')
    plt.title(title)
    
    if sources:
        for source in sources:
            plt.scatter(source[0], source[1], color='red')
    
    plt.show()

def plot_3d_heatmap_cube(qs, x, y, z, sources=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 色範囲の設定
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Reds  # カラーマ}ップを jet に設定（変更可能）

    # 地面 z=0
    ground = qs[0].reshape((x, y))
    X, Y = np.meshgrid(np.linspace(0, x, x), np.linspace(0, y, y))
    ax.plot_surface(X, Y, np.zeros_like(X), facecolors=cmap(norm(ground)), shade=False, zorder=1)

    # 天井 z=10
    ceiling = qs[1].reshape((x, y))
    ax.plot_surface(X, Y, np.full_like(X, z), facecolors=cmap(norm(ceiling)), shade=False, zorder=1)

    # 側面1 x=0
    side1 = qs[2].reshape((y, z))
    Y, Z = np.meshgrid(np.linspace(0, y, y), np.linspace(0, z, z))
    ax.plot_surface(np.zeros_like(Y), Y, Z, facecolors=cmap(norm(side1)), shade=False, zorder=1)

    # 側面2 y=0
    side2 = qs[3].reshape((x, z))
    X, Z = np.meshgrid(np.linspace(0, x, x), np.linspace(0, z, z))
    ax.plot_surface(X, np.zeros_like(X), Z, facecolors=cmap(norm(side2)), shade=False, zorder=1)

    # 側面3 x=10
    side3 = qs[4].reshape((y, z))
    ax.plot_surface(np.full_like(Y, x), Y, Z, facecolors=cmap(norm(side3)), shade=False, zorder=1)

    # 側面4 y=10
    side4 = qs[5].reshape((x, z))
    ax.plot_surface(X, np.full_like(X, y), Z, facecolors=cmap(norm(side4)), shade=False, zorder=1)

    edges = [
    [[x, x], [0, y], [0, 0]],  # edge along y-axis at x=max, z=0
    [[0, x], [y, y], [0, 0]],  # edge along x-axis at y=max, z=0

    [[0, 0], [0, y], [z, z]],  # top edges (z=max)
    [[0, x], [0, 0], [z, z]],
    [[x, x], [0, y], [z, z]],
    [[0, x], [y, y], [z, z]],

    [[x, x], [0, 0], [0, z]],
    [[0, 0], [y, y], [0, z]],
    [[x, x], [y, y], [0, z]],
    ]
    for edge in edges:
        ax.plot(edge[0], edge[1], edge[2], color="black", linewidth=1.5, zorder=100)

    # 放射線源の位置を3D空間内にプロット（赤色の点）
    if sources:
        for source in sources:
            ax.scatter(source[0], source[1], source[2], color='red', s=50)

    # 軸のラベル
    ax.set_xlabel('X [m]', fontsize=20)
    ax.set_ylabel('Y [m]', fontsize=20)
    ax.set_zlabel('Z [m]', fontsize=20)

    # 軸の範囲設定（plot_measurement_points_3d と統一）
    ax.set_xlim(0, x)  # X 軸は左が 0、右が 10
    ax.set_ylim(0, y)  # Y 軸は手前が 0、奥が 10
    ax.set_zlim(0, z)  # Z 軸は通常通り

    # 視点の設定（plot_measurement_points_3d と統一）
    ax.view_init(elev=30, azim=30, roll=0)

    # カラーバーを追加
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.4, aspect=10)
    cbar.set_label('[Bq]', fontsize=14)

    plt.title("3D Heatmap", fontsize=16)
    plt.show()

def plot_heatmap_result_no_colorbar(q, shape, extent, sources=None, vmin=None, vmax=None):
    q_grid = q.reshape(shape)
    plt.figure()
    plt.imshow(q_grid, origin='lower', cmap='Greys', extent=extent, vmin=vmin, vmax=vmax)
    
    # 軸とカラーバーを表示しない
    plt.axis('off')
    
    # 放射線源を表示するオプション
    if sources:
        for source in sources:
            plt.scatter(source[0], source[1], color='red')
    
    plt.show()

def get_transparent_colors(q_values, norm, cmap="Greys", alpha_min=0.1, alpha_max=1.0, threshold=0.2, delta=0.02):
    """透明度を考慮したカラーマップを生成する
    閾値以下は線形補間で透過度を調整し、閾値以上ではさらに少しだけ透過するようにする。
    具体的には、normalized_values < threshold の場合は alpha_min ～ alpha_max の範囲で補間し、
    normalized_values >= threshold の場合は、alpha_max から delta 分だけ減少させる。
    """
    color_map = plt.get_cmap(cmap)
    normalized_values = np.clip(norm(q_values), 0, 1)  # 正規化後、[0, 1]にクリップ
    colors = color_map(normalized_values)  # カラーマップ適用
    
    alphas = np.where(
        normalized_values < threshold,
        alpha_min + (normalized_values / threshold) * (alpha_max - alpha_min),
        # 閾値以上の場合、normalized_values が threshold から 1 にかけて alpha_max から (alpha_max - delta) に補間
        alpha_max - ((normalized_values - threshold) / (1 - threshold)) * delta
    )
    colors[..., -1] = alphas  # 透過度 (alpha) を上書き
    return colors


def plot_3d_heatmap(qs, x, y, z, sources=None, vmin=None, vmax=None):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=30, roll=0)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- XY 平面（地面・天井） ---
    # グリッドを境界（0～x, 0～y）の頂点として作成（形状: (y+1, x+1)）
    Xb, Yb = np.meshgrid(np.linspace(0, x, x+1), np.linspace(0, y, y+1))
    
    # qs[0] は (y,x) のセル中心の値なので、右端・上端をパディングして (y+1,x+1) にする
    ground = qs[0].reshape((y, x))
    ground_pad = np.pad(ground, ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(
        Xb, Yb, np.zeros_like(Xb),  # 地面は z=0
        facecolors=get_transparent_colors(ground_pad, norm, cmap="Greys"),
        shade=False, zorder=1
    )
    ceiling = qs[1].reshape((y, x))
    ceiling_pad = np.pad(ceiling, ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(
        Xb, Yb, np.full_like(Xb, z),  # 天井は z = z
        facecolors=get_transparent_colors(ceiling_pad, norm, cmap="Greys"),
        shade=False, zorder=1
    )
    
    # --- YZ 平面（側面：x=0, x=x） ---
    # グリッド: y方向 0～y, z方向 0～z（形状: (y+1, z+1)）
    Yb_side, Zb_side = np.meshgrid(np.linspace(0, y, y+1), np.linspace(0, z, z+1))
    side1 = qs[2].reshape((z, y))
    side1_pad = np.pad(side1, ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(
        np.full_like(Yb_side, 0),  # x = 0
        Yb_side, Zb_side,
        facecolors=get_transparent_colors(side1_pad, norm, cmap="Greys"),
        shade=False, zorder=1
    )
    side3 = qs[4].reshape((z, y))
    side3_pad = np.pad(side3, ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(
        np.full_like(Yb_side, x),  # x = x
        Yb_side, Zb_side,
        facecolors=get_transparent_colors(side3_pad, norm, cmap="Greys"),
        shade=False, zorder=1
    )
    
    # --- XZ 平面（側面：y=0, y=y） ---
    # グリッド: x方向 0～x, z方向 0～z（形状: (x+1, z+1)）
    Xb_side, Zb_side = np.meshgrid(np.linspace(0, x, x+1), np.linspace(0, z, z+1))
    side2 = qs[3].reshape((z, x))
    side2_pad = np.pad(side2, ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(
        Xb_side, np.full_like(Xb_side, 0),  # y = 0
        Zb_side,
        facecolors=get_transparent_colors(side2_pad, norm, cmap="Greys"),
        shade=False, zorder=1
    )
    side4 = qs[5].reshape((z, x))
    side4_pad = np.pad(side4, ((0, 1), (0, 1)), mode='edge')
    ax.plot_surface(
        Xb_side, np.full_like(Xb_side, y),  # y = y
        Zb_side,
        facecolors=get_transparent_colors(side4_pad, norm, cmap="Greys"),
        shade=False, zorder=1
    )

    # --- 以下はグリッド、エッジ、ソースの描画（物理境界に合わせる） ---
    step = 1
    lw = 0.5
    for i in np.linspace(0, x, x+1):
        ax.plot([i, i], [0, y], [z, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for j in np.linspace(0, y, y+1):
        ax.plot([0, x], [j, j], [z, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for k in np.linspace(0, z, z+1):
        ax.plot([x, x], [0, y], [k, k], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for j in np.linspace(0, y, y+1):
        ax.plot([x, x], [j, j], [0, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for k in np.linspace(0, z, z+1):
        ax.plot([0, x], [y, y], [k, k], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for i in np.linspace(0, x, x+1):
        ax.plot([i, i], [y, y], [0, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)

    if sources:
        for source in sources:
            z_pos = source.z if source.z > 0 else 0.1
            ax.scatter(
                source.x, source.y, z_pos,
                color='#ff0000', s=100,
                edgecolors='black', linewidth=1.5,
                zorder=50
            )
            ax.text(
                source.x+1.5, source.y, z_pos+2.0,
                f'({source.x:.2f}, {source.y:.2f}, {source.z:.2f})',
                color='black', zorder=100, fontsize=20, fontweight='bold', family="serif"
            )
            ax.plot([0, source.x], [0, source.y], [0, z_pos], linestyle="dotted", color='#ff0000', linewidth=2.0, zorder=50)

    edges = [
        [[0, 0], [0, y], [0, 0]],
        [[0, x], [0, 0], [0, 0]],
        [[x, x], [0, y], [0, 0]],
        [[0, x], [y, y], [0, 0]],
        [[0, 0], [0, y], [z, z]],
        [[0, x], [0, 0], [z, z]],
        [[x, x], [0, y], [z, z]],
        [[0, x], [y, y], [z, z]],
        [[0, 0], [0, 0], [0, z]],
        [[x, x], [0, 0], [0, z]],
        [[0, 0], [y, y], [0, z]],
        [[x, x], [y, y], [0, z]],
    ]
    for edge in edges:
        ax.plot(edge[0], edge[1], edge[2], color="black", linewidth=1, zorder=2)

    fontdict = {"fontsize": 20, "family": "serif"}
    ax.set_xlabel('X', fontdict=fontdict)
    ax.set_ylabel('Y', fontdict=fontdict)
    ax.set_zlabel('Z', fontdict=fontdict)

    ax.set_xticks(np.arange(0, x+1, 2))
    ax.set_yticks(np.arange(0, y+1, 2))
    ax.set_zticks(np.arange(0, z+1, 2))
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap='Greys'),
        ax=ax, shrink=0.4, aspect=20, label='[Bq]'
    )
    cbar.ax.set_ylabel("[Bq]", fontdict=fontdict)

    plt.tight_layout()
    plt.show()

