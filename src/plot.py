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
    plt.colorbar(label='[Sv/h]')
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