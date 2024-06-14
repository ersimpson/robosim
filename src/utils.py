"""Module with helper functions for common 2d vector operations."""

import numpy as np


def rad_to_deg(theta: float) -> float:
    """Converts an angle in radians to degrees.

    Args:
        theta (float): an angle in radians.

    Returns:
        A float of an angle in degrees.
    """
    return theta * 180 / np.pi


def get_angle_between_vectors(
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """Get the angle between two vectors.

    Args:
        u (np.ndarray): the first vector.
        v (np.ndarray): the second vector.

    Returns:
        A float of the angle between the two vectors.
    """
    if np.allclose(u, v):
        return 0.0
    
    unorm = np.linalg.norm(u)
    vnorm = np.linalg.norm(v)
    dp = np.dot(u, v) / (unorm * vnorm)
    cp = np.cross(u, v)
    
    ang = np.arccos(dp)
    if cp < 0:
        ang = 2 * np.pi - ang
    return ang


def is_point_in_radius(
    x: float,
    y: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """Check if a point is within a given radius of a center point.

    Args:
        x (float): the x-coordinate of the point.
        y (float): the y-coordinate of the point.
        cx (float): the x-coordinate of the center point.
        cy (float): the y-coordinate of the center point.
        radius (float): the radius to check.

    Returns:
        A bool indicating if the point is within the radius.
    """
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius


def get_local_to_world_orientation(
    lx: float,
    ly: float,
    ltheta: float,
    wx: float,
    wy: float,
) -> float:    
    """Compute the angle between a local orientation and an arbitrary world
    position in radians.

    Args:
        lx (float): the local x-coordinate.
        ly (float): the local y-coordinate.
        ltheta (float): the local orientation in radians.
        wx (float): the world x-coordinate.
        wy (float): the world y-coordinate.

    Returns:
        A float of the world angle in radians.
    """
    lv = np.array([lx, ly])
    wv = np.array([wx, wy])

    # Get vector along robot orientation
    dv = wv - lv
    d = np.linalg.norm(dv)
    ov = np.array([lx + d * np.cos(ltheta), ly + d * np.sin(ltheta)])
    ov = ov - lv
    
    # Find the angle between the local orientation vector and the delta vector
    ang = get_angle_between_vectors(ov, dv)
    return ang
