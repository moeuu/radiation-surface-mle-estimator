import random

import numpy as np

from .config import GeometryConfig, RadiationSource


SHIELD_ATTENUATION = 0.10177304964539008


class radiation_source(RadiationSource):
    pass


def _coerce_source(source):
    if isinstance(source, RadiationSource):
        return source
    return RadiationSource(*source)


def decide_measurement_points(x_or_geometry, y=None, r=None, seed=None, z_level=None):
    if isinstance(x_or_geometry, GeometryConfig):
        geometry = x_or_geometry
        measurement_ratio = y
        if measurement_ratio is None:
            raise ValueError("measurement_ratio is required when passing a GeometryConfig.")
    else:
        if y is None or r is None:
            raise ValueError("x, y, and r are required when passing scalar dimensions.")
        geometry = GeometryConfig(x=float(x_or_geometry), y=float(y), z=1.0, g=1.0)
        measurement_ratio = r

    detector_height = geometry.g / 2 if z_level is None else z_level
    candidates = []
    for i in range(geometry.x_cells):
        for j in range(geometry.y_cells):
            candidates.append([geometry.g / 2 + i * geometry.g, geometry.g / 2 + j * geometry.g, detector_height])

    sample_count = int(measurement_ratio * len(candidates))
    if measurement_ratio > 0 and sample_count == 0:
        sample_count = 1

    rng = random.Random(seed)
    return rng.sample(candidates, sample_count)


def add_shield(m_p, labels=("A", "B", "C", "D")):
    measurement_points_with_shield = []
    for point in m_p:
        for label in labels:
            measurement_points_with_shield.append([*point, label])
    return measurement_points_with_shield


def cartesian_to_spherical(vector):
    x, y, z = np.asarray(vector, dtype=float)
    radius = np.sqrt(x**2 + y**2 + z**2)
    if radius == 0:
        return 0.0, 0.0

    theta = np.arccos(z / radius)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    return theta, phi


def shield_blocks_radiation(shield_orientation, source_position, detector_position):
    vector = np.asarray(source_position, dtype=float) - np.asarray(detector_position, dtype=float)
    theta, phi = cartesian_to_spherical(vector)

    shield_theta_min, shield_theta_max = shield_orientation["theta"]
    shield_phi_min, shield_phi_max = shield_orientation["phi"]
    return shield_theta_min <= theta <= shield_theta_max and shield_phi_min <= phi <= shield_phi_max


def calculate_distance(x1, y1, z1, x2, y2, z2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))


def calculate_attenuation(shield_orientation, source_position, detector_position):
    distance = np.linalg.norm(np.asarray(source_position, dtype=float) - np.asarray(detector_position, dtype=float))
    if distance == 0:
        return 0.0

    attenuation = 1.0 / (distance**2)
    if shield_blocks_radiation(shield_orientation, source_position, detector_position):
        attenuation *= SHIELD_ATTENUATION
    return float(attenuation)


def measurement_shield(m_p, source_list, shield_orientations):
    rad_all_measurements = []
    sources = [_coerce_source(source) for source in source_list]

    for detector_position in m_p:
        for orientation in shield_orientations:
            total_radiation = 0.0
            for source in sources:
                total_radiation += source.intensity * calculate_attenuation(
                    orientation,
                    source.position,
                    detector_position,
                )
            rad_all_measurements.append(total_radiation)

    return rad_all_measurements


def measurement_mle(m_p, source_list):
    rad_all_measurements = []
    sources = [_coerce_source(source) for source in source_list]

    for detector_position in m_p:
        total_radiation = 0.0
        for source in sources:
            distance = calculate_distance(*detector_position, *source.position)
            attenuation = 0.0 if distance == 0 else 1.0 / (distance**2)
            total_radiation += source.intensity * attenuation
        rad_all_measurements.append(total_radiation)

    return rad_all_measurements


def create_A(l1, l2, m_p, grid, shield_orientations, g=1.0):
    expected_columns = int(round(l1 / g)) * int(round(l2 / g))
    if len(grid) != expected_columns:
        raise ValueError(f"Grid size mismatch: expected {expected_columns} points, got {len(grid)}.")

    row_count = len(m_p) * len(shield_orientations)
    matrix = np.zeros((row_count, len(grid)))

    for point_index, detector_position in enumerate(m_p):
        for orientation_index, orientation in enumerate(shield_orientations):
            row_index = point_index * len(shield_orientations) + orientation_index
            for grid_index, grid_position in enumerate(grid):
                matrix[row_index, grid_index] = calculate_attenuation(
                    orientation,
                    grid_position,
                    detector_position,
                )

    return matrix


def create_A_mle(l1, l2, m_p, grid, g=1.0):
    expected_columns = int(round(l1 / g)) * int(round(l2 / g))
    if len(grid) != expected_columns:
        raise ValueError(f"Grid size mismatch: expected {expected_columns} points, got {len(grid)}.")

    matrix = np.zeros((len(m_p), len(grid)))
    for point_index, detector_position in enumerate(m_p):
        for grid_index, grid_position in enumerate(grid):
            distance = calculate_distance(*detector_position, *grid_position)
            matrix[point_index, grid_index] = 0.0 if distance == 0 else 1.0 / (distance**2)
    return matrix


def find_nearest_grid(source, grid):
    nearest_grid_point = None
    nearest_grid_index = -1
    min_distance = float("inf")

    for index, point in enumerate(grid):
        distance = calculate_distance(source[0], source[1], source[2], point[0], point[1], point[2])
        if distance < min_distance:
            min_distance = distance
            nearest_grid_index = index
            nearest_grid_point = point

    return nearest_grid_point, nearest_grid_index


def assign_source_intensity_for_check(source, grids, q_check_vectors):
    source_obj = _coerce_source(source)
    nearest_grid_index = -1
    nearest_q_vector = None
    min_distance = float("inf")

    for grid, q_check_vector in zip(grids, q_check_vectors):
        for index, point in enumerate(grid):
            distance = calculate_distance(*source_obj.position, *point)
            if distance < min_distance:
                min_distance = distance
                nearest_grid_index = index
                nearest_q_vector = q_check_vector

    if nearest_q_vector is not None and nearest_grid_index != -1:
        nearest_q_vector.fill(0)
        nearest_q_vector[nearest_grid_index] = source_obj.intensity

    return q_check_vectors
