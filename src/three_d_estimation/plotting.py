import matplotlib.pyplot as plt
import numpy as np


def _coerce_source(source):
    if hasattr(source, "intensity"):
        return source.x, source.y, source.z, source.intensity
    if hasattr(source, "bq"):
        return source.x, source.y, source.z, source.bq
    return source[0], source[1], source[2], source[3]


def plot_measurement_points(m_points, sources, title, x, y):
    x_coords, y_coords = zip(*[(point[0], point[1]) for point in m_points])
    plt.figure()
    plt.scatter(x_coords, y_coords, color="blue", label="Measurement points")

    for index, source in enumerate(sources or []):
        source_x, source_y, _, _ = _coerce_source(source)
        plt.scatter(source_x, source_y, color="red", marker="x", label="Sources" if index == 0 else None)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(0, x)
    plt.ylim(0, y)
    plt.grid(True, linestyle="--", linewidth=1, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def plot_measurement_points_3d(m_points, title, x, y, z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if m_points:
        x_coords, y_coords, z_coords = zip(*m_points)
        ax.scatter(x_coords, y_coords, z_coords, color="blue", label="Measurement points")
    ax.set_title(title)
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(q, shape, extent, title, sources=None, vmin=None, vmax=None):
    q_grid = np.asarray(q).reshape(shape)
    plt.figure()
    plt.imshow(q_grid, origin="lower", cmap="Greys", extent=extent, vmin=vmin, vmax=vmax)
    plt.colorbar(label="[Bq]")
    plt.title(title)

    if sources:
        for source in sources:
            source_x, source_y, _, _ = _coerce_source(source)
            plt.scatter(source_x, source_y, color="red")

    plt.tight_layout()
    plt.show()


def plot_heatmap_result_no_colorbar(q, shape, extent, sources=None, vmin=None, vmax=None):
    q_grid = np.asarray(q).reshape(shape)
    plt.figure()
    plt.imshow(q_grid, origin="lower", cmap="Greys", extent=extent, vmin=vmin, vmax=vmax)
    plt.axis("off")

    if sources:
        for source in sources:
            source_x, source_y, _, _ = _coerce_source(source)
            plt.scatter(source_x, source_y, color="red")

    plt.tight_layout()
    plt.show()


def get_transparent_colors(q_values, norm, cmap="Greys", alpha_min=0.1, alpha_max=1.0, threshold=0.2, delta=0.02):
    color_map = plt.get_cmap(cmap)
    normalized_values = np.clip(norm(q_values), 0, 1)
    colors = color_map(normalized_values)
    alphas = np.where(
        normalized_values < threshold,
        alpha_min + (normalized_values / threshold) * (alpha_max - alpha_min),
        alpha_max - ((normalized_values - threshold) / (1 - threshold)) * delta,
    )
    colors[..., -1] = alphas
    return colors


def _plot_surface(ax, face_values, coordinates, norm, cmap):
    padded_values = np.pad(face_values, ((0, 1), (0, 1)), mode="edge")
    ax.plot_surface(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        facecolors=get_transparent_colors(padded_values, norm, cmap=cmap),
        shade=False,
        zorder=1,
    )


def plot_3d_heatmap_cube(qs, x, y, z, sources=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=30, roll=0)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = "Reds"

    floor = np.asarray(qs[0])
    side_yz = np.asarray(qs[2])
    side_xz = np.asarray(qs[3])

    x_grid, y_grid = np.meshgrid(
        np.linspace(0, x, floor.shape[1] + 1),
        np.linspace(0, y, floor.shape[0] + 1),
    )
    yz_y, yz_z = np.meshgrid(
        np.linspace(0, y, side_yz.shape[1] + 1),
        np.linspace(0, z, side_yz.shape[0] + 1),
    )
    xz_x, xz_z = np.meshgrid(
        np.linspace(0, x, side_xz.shape[1] + 1),
        np.linspace(0, z, side_xz.shape[0] + 1),
    )

    _plot_surface(ax, floor, (x_grid, y_grid, np.zeros_like(x_grid)), norm, cmap)
    _plot_surface(ax, np.asarray(qs[1]), (x_grid, y_grid, np.full_like(x_grid, z)), norm, cmap)
    _plot_surface(ax, side_yz, (np.zeros_like(yz_y), yz_y, yz_z), norm, cmap)
    _plot_surface(ax, side_xz, (xz_x, np.zeros_like(xz_x), xz_z), norm, cmap)
    _plot_surface(ax, np.asarray(qs[4]), (np.full_like(yz_y, x), yz_y, yz_z), norm, cmap)
    _plot_surface(ax, np.asarray(qs[5]), (xz_x, np.full_like(xz_x, y), xz_z), norm, cmap)

    if sources:
        for source in sources:
            source_x, source_y, source_z, _ = _coerce_source(source)
            ax.scatter(source_x, source_y, source_z, color="red", s=50, zorder=20)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.4, aspect=12)
    cbar.set_label("[Bq]")

    plt.tight_layout()
    plt.show()


def plot_3d_heatmap(qs, x, y, z, sources=None, vmin=None, vmax=None):
    plot_3d_heatmap_cube(qs, x, y, z, sources=sources, vmin=vmin, vmax=vmax)
