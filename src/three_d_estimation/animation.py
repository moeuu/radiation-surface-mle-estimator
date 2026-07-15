import numpy as np


def build_sphere_mesh(radius, resolution=100):
    """Return Cartesian arrays for a spherical visualization mesh."""
    azimuth = np.linspace(0, 2 * np.pi, resolution)
    polar = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(azimuth), np.sin(polar))
    y = radius * np.outer(np.sin(azimuth), np.sin(polar))
    z = radius * np.outer(np.ones(np.size(azimuth)), np.cos(polar))
    return x, y, z


def build_shell_mesh(theta_range, phi_range, radius, resolution=50):
    """Return Cartesian arrays for one angular shield-shell section."""
    phi = np.linspace(phi_range[0], phi_range[1], resolution)
    theta = np.linspace(theta_range[0], theta_range[1], resolution)
    x = radius * np.outer(np.cos(phi), np.sin(theta))
    y = radius * np.outer(np.sin(phi), np.sin(theta))
    z = radius * np.outer(np.ones(np.size(phi)), np.cos(theta))
    return x, y, z


def draw_shield(
    ax,
    orientation,
    inner_radius=9,
    outer_radius=10,
    source_position=None,
    frame_number=None,
    base_color="blue",
    shield_color="cyan",
):
    """Render one rotating-shield frame on a Matplotlib 3-D axis."""
    ax.cla()
    x_full, y_full, z_full = build_sphere_mesh(inner_radius)
    ax.plot_surface(x_full, y_full, z_full, color=base_color, alpha=0.6, edgecolor="gray")

    theta_range = orientation["theta"]
    phi_range = orientation["phi"]
    x_outer, y_outer, z_outer = build_shell_mesh(theta_range, phi_range, outer_radius)
    x_inner, y_inner, z_inner = build_shell_mesh(theta_range, phi_range, inner_radius)
    ax.plot_surface(x_outer, y_outer, z_outer, color=shield_color, alpha=0.3, edgecolor="gray")
    ax.plot_surface(x_inner, y_inner, z_inner, color="white", alpha=0.3, edgecolor="gray")

    if source_position is not None:
        ax.scatter([source_position[0]], [source_position[1]], [source_position[2]], color="red", s=100, label="Radiation Source")

    if frame_number is not None:
        ax.text2D(0.05, 0.9, f"Frame: {frame_number + 1}", transform=ax.transAxes, color="black")

    ax.set_xlim([-outer_radius, outer_radius])
    ax.set_ylim([-outer_radius, outer_radius])
    ax.set_zlim([-outer_radius, outer_radius])
    ax.view_init(elev=30, azim=45)
    return ax
