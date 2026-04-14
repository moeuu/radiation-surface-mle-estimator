import numpy as np

from .config import GeometryConfig


def create_grid(axis, position, g=1.0, x=10.0, y=10.0, z=10.0):
    geometry = GeometryConfig(x=float(x), y=float(y), z=float(z), g=float(g))
    grid = []

    if axis == "x":
        for z_index in range(geometry.z_cells):
            for y_index in range(geometry.y_cells):
                grid.append([position, geometry.g / 2 + y_index * geometry.g, geometry.g / 2 + z_index * geometry.g])
    elif axis == "y":
        for z_index in range(geometry.z_cells):
            for x_index in range(geometry.x_cells):
                grid.append([geometry.g / 2 + x_index * geometry.g, position, geometry.g / 2 + z_index * geometry.g])
    elif axis == "z":
        for y_index in range(geometry.y_cells):
            for x_index in range(geometry.x_cells):
                grid.append([geometry.g / 2 + x_index * geometry.g, geometry.g / 2 + y_index * geometry.g, position])
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    return grid


def create_face_grids(geometry):
    return [
        create_grid("z", 0.0, g=geometry.g, x=geometry.x, y=geometry.y, z=geometry.z),
        create_grid("z", geometry.z, g=geometry.g, x=geometry.x, y=geometry.y, z=geometry.z),
        create_grid("x", 0.0, g=geometry.g, x=geometry.x, y=geometry.y, z=geometry.z),
        create_grid("y", 0.0, g=geometry.g, x=geometry.x, y=geometry.y, z=geometry.z),
        create_grid("x", geometry.x, g=geometry.g, x=geometry.x, y=geometry.y, z=geometry.z),
        create_grid("y", geometry.y, g=geometry.g, x=geometry.x, y=geometry.y, z=geometry.z),
    ]


def get_face_array_shapes(geometry):
    return (
        (geometry.y_cells, geometry.x_cells),
        (geometry.y_cells, geometry.x_cells),
        (geometry.z_cells, geometry.y_cells),
        (geometry.z_cells, geometry.x_cells),
        (geometry.z_cells, geometry.y_cells),
        (geometry.z_cells, geometry.x_cells),
    )


def get_face_vector_shapes(geometry):
    return tuple((rows * cols, 1) for rows, cols in get_face_array_shapes(geometry))


def build_initial_q(geometry, q_max):
    q_init = 1.0 / q_max
    return np.vstack([np.full(shape, q_init, dtype=float) for shape in get_face_vector_shapes(geometry)])


def restore_q(q_optimized, q_shapes):
    restored_q = []
    index = 0
    flat_q = np.asarray(q_optimized, dtype=float).reshape(-1, 1)
    for shape in q_shapes:
        length = int(np.prod(shape))
        restored_q.append(flat_q[index : index + length].reshape(shape))
        index += length
    return restored_q


def get_grid_position(surface_idx, grid_idx, x, y, z, g=1.0):
    geometry = GeometryConfig(x=float(x), y=float(y), z=float(z), g=float(g))

    if surface_idx == 0:
        return np.array([geometry.g / 2 + (grid_idx % geometry.x_cells) * geometry.g, geometry.g / 2 + (grid_idx // geometry.x_cells) * geometry.g, 0.0])
    if surface_idx == 1:
        return np.array([geometry.g / 2 + (grid_idx % geometry.x_cells) * geometry.g, geometry.g / 2 + (grid_idx // geometry.x_cells) * geometry.g, geometry.z])
    if surface_idx == 2:
        return np.array([0.0, geometry.g / 2 + (grid_idx % geometry.y_cells) * geometry.g, geometry.g / 2 + (grid_idx // geometry.y_cells) * geometry.g])
    if surface_idx == 3:
        return np.array([geometry.g / 2 + (grid_idx % geometry.x_cells) * geometry.g, 0.0, geometry.g / 2 + (grid_idx // geometry.x_cells) * geometry.g])
    if surface_idx == 4:
        return np.array([geometry.x, geometry.g / 2 + (grid_idx % geometry.y_cells) * geometry.g, geometry.g / 2 + (grid_idx // geometry.y_cells) * geometry.g])
    if surface_idx == 5:
        return np.array([geometry.g / 2 + (grid_idx % geometry.x_cells) * geometry.g, geometry.y, geometry.g / 2 + (grid_idx // geometry.x_cells) * geometry.g])
    raise ValueError(f"Invalid surface index: {surface_idx}")


def split_q_into_faces(q, x, y, z, g=1.0):
    geometry = GeometryConfig(x=float(x), y=float(y), z=float(z), g=float(g))
    flat_q = np.asarray(q, dtype=float).flatten()
    sizes = [rows * cols for rows, cols in get_face_array_shapes(geometry)]
    faces = []
    start = 0
    for size in sizes:
        faces.append(flat_q[start : start + size])
        start += size
    return faces


def compress_q_by_local_max(q, x, y, z, g=1.0):
    geometry = GeometryConfig(x=float(x), y=float(y), z=float(z), g=float(g))
    q_splitted = split_q_into_faces(q, geometry.x, geometry.y, geometry.z, geometry.g)
    q_compressed = []

    for face, shape in zip(q_splitted, get_face_array_shapes(geometry)):
        grid = face.reshape(shape)
        padded_grid = np.zeros((grid.shape[0] + 6, grid.shape[1] + 6))
        padded_grid[3:-3, 3:-3] = grid
        original = padded_grid.copy()

        for row_index in range(3, padded_grid.shape[0] - 3):
            for column_index in range(3, padded_grid.shape[1] - 3):
                sub_grid = padded_grid[row_index - 3 : row_index + 4, column_index - 3 : column_index + 4]
                max_index = np.unravel_index(np.argmax(sub_grid), sub_grid.shape)
                if padded_grid[row_index, column_index] == sub_grid[max_index]:
                    padded_grid[row_index, column_index] += np.sum(sub_grid)

        padded_grid -= original
        q_compressed.append(padded_grid[3:-3, 3:-3].reshape(-1, 1))

    return np.vstack(q_compressed)


def get_nonzero_coords_and_values(qs_split, x, y, z, threshold=1.0, g=1.0):
    geometry = GeometryConfig(x=float(x), y=float(y), z=float(z), g=float(g))
    results = []

    for surface_idx, (grid, shape) in enumerate(zip(qs_split, get_face_array_shapes(geometry))):
        surface = np.asarray(grid, dtype=float).reshape(shape)
        for row_index in range(shape[0]):
            for column_index in range(shape[1]):
                value = surface[row_index, column_index]
                if abs(value) <= threshold:
                    continue

                if surface_idx == 0:
                    coords = (geometry.g / 2 + column_index * geometry.g, geometry.g / 2 + row_index * geometry.g, 0.0)
                elif surface_idx == 1:
                    coords = (geometry.g / 2 + column_index * geometry.g, geometry.g / 2 + row_index * geometry.g, geometry.z)
                elif surface_idx == 2:
                    coords = (0.0, geometry.g / 2 + column_index * geometry.g, geometry.g / 2 + row_index * geometry.g)
                elif surface_idx == 3:
                    coords = (geometry.g / 2 + column_index * geometry.g, 0.0, geometry.g / 2 + row_index * geometry.g)
                elif surface_idx == 4:
                    coords = (geometry.x, geometry.g / 2 + column_index * geometry.g, geometry.g / 2 + row_index * geometry.g)
                else:
                    coords = (geometry.g / 2 + column_index * geometry.g, geometry.y, geometry.g / 2 + row_index * geometry.g)
                results.append((*coords, value))

    return results
