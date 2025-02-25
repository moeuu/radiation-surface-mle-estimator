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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 色範囲の設定
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # 地面 z=0
    ground = qs[0].reshape((x, y))
    X, Y = np.meshgrid(np.linspace(0, x, x), np.linspace(0, y, y))
    ax.plot_surface(X, Y, np.zeros_like(X), facecolors=plt.cm.Greys(norm(ground)), shade=False)

    # 天井 z=10
    ceiling = qs[1].reshape((x, y))
    ax.plot_surface(X, Y, np.full_like(X, z), facecolors=plt.cm.Greys(norm(ceiling)), shade=False)

    # 側面1 x=0
    side1 = qs[2].reshape((y, z))
    Y, Z = np.meshgrid(np.linspace(0, y, y), np.linspace(0, z, z))
    ax.plot_surface(np.zeros_like(Y), Y, Z, facecolors=plt.cm.Greys(norm(side1)), shade=False)

    # 側面2 y=0
    side2 = qs[3].reshape((x, z))
    X, Z = np.meshgrid(np.linspace(0, x, x), np.linspace(0, z, z))
    ax.plot_surface(X, np.zeros_like(X), Z, facecolors=plt.cm.Greys(norm(side2)), shade=False)

    # 側面3 x=10
    side3 = qs[4].reshape((y, z))
    ax.plot_surface(np.full_like(Y, x), Y, Z, facecolors=plt.cm.Greys(norm(side3)), shade=False)

    # 側面4 y=10
    side4 = qs[5].reshape((x, z))
    ax.plot_surface(X, np.full_like(X, y), Z, facecolors=plt.cm.Greys(norm(side4)), shade=False)

    # 放射線源の位置を3D空間内にプロット（赤色の点）
    if sources:
        for source in sources:
            ax.scatter(source[0], source[1], source[2], color='red', s=50)

    # 軸のラベル
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # カラーバーの範囲のみ指定（表示しない）
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Greys'), ax=ax, shrink=0.5, aspect=5, label='[Bq]')
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

def get_transparent_colors(q_values, norm, cmap="Greys", alpha_min=0.1, alpha_max=1.0):
    """透明度を考慮したカラーマップを生成する"""
    color_map = plt.get_cmap(cmap)
    normalized_values = norm(q_values)  # 正規化
    colors = color_map(normalized_values)  # カラーマップ適用
    colors[..., -1] = alpha_min + (normalized_values * (alpha_max - alpha_min))  # 透過度調整
    return colors

def plot_3d_heatmap(qs, x, y, z, sources=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=30, roll=0)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # **XY 平面 (z=0, z=10)**
    X, Y = np.meshgrid(np.arange(x), np.arange(y))

    # **地面 z=0**
    ground = qs[0].reshape((y, x))
    ax.plot_surface(X, Y, np.zeros_like(X), facecolors=get_transparent_colors(ground, norm, cmap="Greys"), shade=False, zorder=1)

    # **天井 z=10**
    ceiling = qs[1].reshape((y, x))
    ax.plot_surface(X, Y, np.full_like(X, z), facecolors=get_transparent_colors(ceiling, norm, cmap="Greys"), shade=False, zorder=1)

    # **YZ 平面 (x=0, x=10)**
    Y, Z = np.meshgrid(np.arange(y), np.arange(z))

    # **側面1 x=0**
    side1 = qs[2].reshape((z, y))
    ax.plot_surface(np.zeros_like(Y), Y, Z, facecolors=get_transparent_colors(side1, norm, cmap="Greys"), shade=False, zorder=1)

    # **側面3 x=10**
    side3 = qs[4].reshape((z, y))
    ax.plot_surface(np.full_like(Y, x), Y, Z, facecolors=get_transparent_colors(side3, norm, cmap="Greys"), shade=False, zorder=1)

    # **XZ 平面 (y=0, y=10)**
    X, Z = np.meshgrid(np.arange(x), np.arange(z))

    # **側面2 y=0**
    side2 = qs[3].reshape((z, x))
    ax.plot_surface(X, np.zeros_like(X), Z, facecolors=get_transparent_colors(side2, norm, cmap="Greys"), shade=False, zorder=1)

    # **側面4 y=10**
    side4 = qs[5].reshape((z, x))
    ax.plot_surface(X, np.full_like(X, y), Z, facecolors=get_transparent_colors(side4, norm, cmap="Greys"), shade=False, zorder=1)

    # **点線の復元**
    step = 2  # 2ごとに点線を描画
    lw = 0.5  # 極細の線

    for i in range(0, x+1, step):
        ax.plot([i, i], [0, y], [z, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for j in range(0, y+1, step):
        ax.plot([0, x], [j, j], [z, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)

    for k in range(0, z+1, step):
        ax.plot([x, x], [0, y], [k, k], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for j in range(0, y+1, step):
        ax.plot([x, x], [j, j], [0, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)

    for k in range(0, z+1, step):
        ax.plot([0, x], [y, y], [k, k], linestyle="dotted", color="black", linewidth=lw, zorder=2)
    for i in range(0, x+1, step):
        ax.plot([i, i], [y, y], [0, z], linestyle="dotted", color="black", linewidth=lw, zorder=2)

    # **放射線源のプロット**
    if sources:
        for source in sources:
            z_pos = source.z if source.z > 0 else 0.1 

            ax.scatter(source.x, source.y, z_pos, color='#ff0000', s=100, edgecolors='black', linewidth=1.5, facecolors='red', zorder=50)
            ax.text(source.x, source.y, z_pos, f'({source.x:.2f}, {source.y:.2f}, {source.z:.2f})', 
                    color='black', zorder=50, fontsize=10, fontweight='bold', family="serif")
            ax.plot([0, source.x], [0, source.y], [0, z_pos], color='#ff0000', linewidth=3, zorder=50)

    # **枠線の復元（すべての辺の太さを統一）**
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

    # **軸設定（フォント修正）**
    fontdict = {"fontsize": 12, "family": "serif"}
    ax.set_xlabel('X', fontdict=fontdict)
    ax.set_ylabel('Y', fontdict=fontdict)
    ax.set_zlabel('Z', fontdict=fontdict)

    ax.set_xticks(np.arange(0, x+1, 2))
    ax.set_yticks(np.arange(0, y+1, 2))
    ax.set_zticks(np.arange(0, z+1, 2))
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)

    # **カラーバーのサイズを調整**
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Greys'), ax=ax, shrink=0.4, aspect=20, label='[Bq]')
    cbar.ax.set_ylabel("[Bq]", fontdict=fontdict)
    
    plt.tight_layout()
    plt.show()