import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from three_d_estimation.animation import draw_shield
from three_d_estimation.config import default_shield_orientations


def main():
    orientations = default_shield_orientations()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        orientation = orientations[frame % len(orientations)]
        draw_shield(ax, orientation, source_position=(10, 10, 10), frame_number=frame)
        return (ax,)

    FuncAnimation(fig, update, frames=len(orientations), interval=1250, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
