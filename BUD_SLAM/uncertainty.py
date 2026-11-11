import numpy as np


def compute_point_uncertainty(depth_conf, world_points=None, world_points_alt=None,
                               conf_weight=0.6, geom_weight=0.4, eps=1e-6):
    """
    Per-pixel uncertainty estimate in [0, 1], higher = less certain.

    Combines two independent signals:
      1. Model confidence (depth_conf): low confidence -> high uncertainty.
         Normalized per-batch via min/max so it stays comparable across
         submaps that may have very different confidence ranges.
      2. Depth-head / point-head disagreement (when world_points_alt is
         given): the same signal already used for dynamic-point detection
         in Solver, reused here as a continuous geometric uncertainty proxy
         instead of a hard threshold.

    Args:
        depth_conf: (S, H, W) model confidence from depth_head.
        world_points: (S, H, W, 3) depth_head 3D points. Only used together
            with world_points_alt for the geometric term.
        world_points_alt: (S, H, W, 3) point_head 3D points, or None.
        conf_weight: weight of the confidence-based term.
        geom_weight: weight of the geometric disagreement term. Ignored
            (and renormalized away) if world_points_alt is None.

    Returns:
        uncertainty: (S, H, W) float32 array in [0, 1].
    """
    depth_conf = np.asarray(depth_conf, dtype=np.float32)

    # --- confidence-based term ---
    conf_min = depth_conf.min()
    conf_max = depth_conf.max()
    if conf_max - conf_min < eps:
        conf_term = np.zeros_like(depth_conf)
    else:
        conf_norm = (depth_conf - conf_min) / (conf_max - conf_min)
        conf_term = 1.0 - conf_norm

    has_geom = world_points is not None and world_points_alt is not None
    if not has_geom:
        return np.clip(conf_term, 0.0, 1.0)

    # --- geometric disagreement term ---
    world_points = np.asarray(world_points, dtype=np.float32)
    world_points_alt = np.asarray(world_points_alt, dtype=np.float32)

    diff = np.linalg.norm(world_points - world_points_alt, axis=-1)  # (S, H, W)
    # Scale disagreement relative to distance from the submap origin so
    # far-away points (naturally noisier) don't dominate near-field points.
    dist_from_origin = np.linalg.norm(world_points, axis=-1) + eps
    relative_disagreement = diff / dist_from_origin

    geom_min = relative_disagreement.min()
    geom_max = relative_disagreement.max()
    if geom_max - geom_min < eps:
        geom_term = np.zeros_like(relative_disagreement)
    else:
        geom_term = (relative_disagreement - geom_min) / (geom_max - geom_min)

    total_weight = conf_weight + geom_weight
    uncertainty = (conf_weight * conf_term + geom_weight * geom_term) / total_weight
    return np.clip(uncertainty, 0.0, 1.0)


def uncertainty_to_color(uncertainty):
    """
    Map an uncertainty array in [0, 1] to RGB uint8 colors, blue (certain)
    -> red (uncertain). No matplotlib dependency so it stays cheap to call
    per-frame.
    """
    u = np.clip(uncertainty, 0.0, 1.0)
    r = (u * 255).astype(np.uint8)
    b = ((1.0 - u) * 255).astype(np.uint8)
    g = np.zeros_like(r)
    return np.stack([r, g, b], axis=-1)
