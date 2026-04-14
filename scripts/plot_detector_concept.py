import matplotlib.pyplot as plt
import numpy as np


def main():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    font_settings = {"fontsize": 16, "weight": "bold"}
    radius_inner = 6
    radius_outer = 6.2
    shift_x = -3

    u = np.linspace(0, np.pi / 2, 40)
    v = np.linspace(0, 2 * np.pi, 40)
    u, v = np.meshgrid(u, v)
    x_sphere = radius_inner * np.sin(u) * np.cos(v) + shift_x
    y_sphere = radius_inner * np.sin(u) * np.sin(v)
    z_sphere = radius_inner * np.cos(u)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color="blue", alpha=0.3, edgecolor="none")

    theta = np.linspace(0, np.pi / 2, 30)
    phi = np.linspace(-np.pi / 2, 0, 30)
    theta, phi = np.meshgrid(theta, phi)
    x_outer = radius_outer * np.sin(theta) * np.cos(phi) + shift_x
    y_outer = radius_outer * np.sin(theta) * np.sin(phi)
    z_outer = radius_outer * np.cos(theta)
    x_inner = radius_inner * np.sin(theta) * np.cos(phi) + shift_x
    y_inner = radius_inner * np.sin(theta) * np.sin(phi)
    z_inner = radius_inner * np.cos(theta)
    ax.plot_surface(x_outer, y_outer, z_outer, color="gray", alpha=0.6, edgecolor="none")
    ax.plot_surface(x_inner, y_inner, z_inner, color="gray", alpha=0.2, edgecolor="none")

    ax.text(5, 3, 6, "Nondirectional Detector", color="blue", ha="left", **font_settings)
    ax.text(5, 3, 5, "Lightweight Shield", color="black", ha="left", **font_settings)
    ax.axis("off")
    ax.set_xlim([-8, 8])
    ax.set_ylim([-6, 6])
    ax.set_zlim([0, 8])
    plt.show()


if __name__ == "__main__":
    main()
