"""
Render either:
  (a) a "zoom in then orbit" video around a point cloud, or
  (b) a first-person "flythrough" video that follows the actual recorded
      camera trajectory (poses.txt from --log_results),
from a point cloud (.pcd/.ply), using Open3D's Filament-based
OffscreenRenderer (headless, no window/X server required). Frames are then
encoded into an .mp4 with ffmpeg.

Room-scale reconstructions from VGGT-SLAM are:
  - noisy/blurry near the outer boundary (crop/filter it out)
  - NOT aligned to gravity/Z by default, since the coordinate frame is tied
    to the first camera pose. Two alignment strategies are available:
      --align_method floor_plane  : RANSAC-fit the dominant plane in the point
                                     cloud and rotate it onto Z. Works well
                                     when the floor is densely reconstructed,
                                     but can pick the wrong plane (e.g. a wall
                                     or table) if the floor has holes / is
                                     sparse.
      --align_method camera_up    : Average the "up" direction of every
                                     recorded camera pose (rotated into world
                                     frame) to estimate gravity directly from
                                     the trajectory, ignoring the point cloud
                                     entirely. More robust when the floor
                                     reconstruction is patchy, since it relies
                                     on hundreds of poses rather than local
                                     point density. Requires --traj_file even
                                     in orbit mode.
      --align_method none         : no alignment.

Usage (orbit mode: outer orbit -> zoom in -> inner orbit):
    python3 render_video.py \
        --pcd outputs/office_loop_points.pcd \
        --out outputs/office_loop_rotation.mp4 \
        --align_method floor_plane \
        --crop_radius 3.0 \
        --remove_outliers \
        --start_radius 6.0 --end_radius 2.0 \
        --outer_orbit_frames 300 --zoom_frames 90 --inner_orbit_frames 300 \
        --elevation_deg 20 \
        --fps 30

Usage (orbit mode, camera-up alignment, for patchy floors):
    python3 render_video.py \
        --pcd outputs/office_loop_points.pcd \
        --out outputs/office_loop_rotation.mp4 \
        --align_method camera_up \
        --traj_file outputs/office_loop.txt \
        --crop_radius 3.0 \
        --remove_outliers \
        --start_radius 6.0 --end_radius 2.0 \
        --outer_orbit_frames 300 --zoom_frames 90 --inner_orbit_frames 300 \
        --elevation_deg 20 \
        --fps 30

Usage (trajectory flythrough mode):
    python3 render_video.py \
        --pcd outputs/apartment_points.pcd \
        --mode trajectory \
        --traj_file outputs/apartment.txt \
        --out outputs/apartment_flythrough.mp4 \
        --align_method camera_up \
        --remove_outliers \
        --traj_duration 20 \
        --look_ahead_dist 0.8 \
        --fps 30
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
import open3d as o3d
from open3d.visualization import rendering


def parse_args():
    parser = argparse.ArgumentParser(description="Render an orbit or trajectory-flythrough point cloud video (headless)")
    parser.add_argument("--pcd", type=str, required=True, help="Path to input point cloud (.pcd/.ply)")
    parser.add_argument("--out", type=str, default="rotation.mp4", help="Output video path")
    parser.add_argument("--mode", type=str, choices=["orbit", "trajectory"], default="orbit", help="orbit: outer orbit -> zoom in -> inner orbit around the cloud. trajectory: fly through the recorded camera path.")

    # --- Alignment ---
    parser.add_argument("--align_method", type=str, choices=["floor_plane", "camera_up", "none"], default="floor_plane",
                         help="floor_plane: RANSAC-fit the dominant plane (best with dense floors). "
                              "camera_up: estimate gravity from averaged camera-up vectors across the trajectory "
                              "(more robust when the floor has holes/is sparse). none: skip alignment.")
    parser.add_argument("--floor_thresh", type=float, default=0.03, help="[floor_plane] RANSAC inlier distance threshold (meters)")
    parser.add_argument("--floor_ransac_n", type=int, default=3, help="[floor_plane] Number of points to sample per RANSAC iteration")
    parser.add_argument("--floor_iterations", type=int, default=1000, help="[floor_plane] Number of RANSAC iterations")
    parser.add_argument("--cam_up_axis", type=str, default="-y", choices=["x", "-x", "y", "-y", "z", "-z"],
                         help="[camera_up] Which local camera axis points 'up' in the camera frame. VGGT/COLMAP-style "
                              "extrinsics use image coordinates (Y down), so local up is -y (the default). Flip this "
                              "if the aligned result comes out upside down.")

    # Cleanup / cropping to remove blurry outer region
    parser.add_argument("--remove_outliers", action="store_true", help="Apply statistical outlier removal before rendering")
    parser.add_argument("--outlier_nb_neighbors", type=int, default=20, help="Neighbors to use for statistical outlier removal")
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0, help="Std ratio threshold for statistical outlier removal (lower = more aggressive)")
    parser.add_argument("--crop_radius", type=float, default=None, help="Keep only points within this radius (meters) of the crop center. Only used in orbit mode; trajectory mode keeps the full cloud since the camera moves through it.")
    parser.add_argument("--crop_center", type=float, nargs=3, default=None, help="Crop center as x y z. Defaults to the point cloud centroid. Applied AFTER alignment, in the aligned frame.")
    parser.add_argument("--voxel_size", type=float, default=None, help="Optional voxel downsample size before rendering")

    # --- Orbit mode camera path: phase 1 = outer orbit, phase 2 = zoom in, phase 3 = inner orbit ---
    parser.add_argument("--start_radius", type=float, default=6.0, help="Camera distance from lookat for the outer orbit (meters)")
    parser.add_argument("--end_radius", type=float, default=2.0, help="Camera distance from lookat for the inner orbit, once zoomed in (meters)")
    parser.add_argument("--outer_orbit_frames", type=int, default=300, help="Number of frames for the outer orbit phase (full 360 degrees)")
    parser.add_argument("--zoom_frames", type=int, default=90, help="Number of frames for the zoom-in phase (transition from outer to inner radius)")
    parser.add_argument("--inner_orbit_frames", type=int, default=300, help="Number of frames for the inner orbit phase (full 360 degrees)")
    parser.add_argument("--elevation_deg", type=float, default=15.0, help="Camera elevation angle above the horizontal plane, degrees (meaningful once alignment is used)")
    parser.add_argument("--start_azimuth_deg", type=float, default=0.0, help="Starting azimuth angle, degrees")

    # --- Trajectory mode ---
    parser.add_argument("--traj_file", type=str, default=None, help="Path to poses.txt (TUM format: frame_id tx ty tz qx qy qz qw). Required for --mode trajectory, and also for --align_method camera_up.")
    parser.add_argument("--traj_duration", type=float, default=20.0, help="Total duration of the flythrough in seconds (frame count = duration * fps)")
    parser.add_argument("--traj_smooth_window", type=int, default=9, help="Moving-average window (in recorded poses) to smooth out jittery SLAM positions. 1 = no smoothing")
    parser.add_argument("--look_ahead_dist", type=float, default=0.8, help="Distance (meters) ahead along the path the camera looks at, for a stable flythrough")
    parser.add_argument("--height_offset", type=float, default=0.0, help="Extra Z offset (meters) added to the camera height along the trajectory, in the aligned frame (positive = up)")
    parser.add_argument("--traj_up", type=float, nargs=3, default=[0.0, 0.0, 1.0], help="World up vector used to keep the camera level during the flythrough")

    parser.add_argument("--fov_deg", type=float, default=60.0, help="Vertical field of view, degrees")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--point_size", type=float, default=3.0, help="Point size in the render")
    parser.add_argument("--bg_color", type=float, nargs=4, default=[0.05, 0.05, 0.05, 1.0], help="Background RGBA, 0-1 range")
    parser.add_argument("--keep_frames_dir", type=str, default=None, help="If set, keep the rendered PNG frames in this directory instead of a temp dir")
    return parser.parse_args()


def spherical_to_cartesian(radius, azimuth_deg, elevation_deg):
    """azimuth measured around +Z axis, elevation above the XY plane. Returns eye position relative to lookat."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return np.array([x, y, z])


def rotation_matrix_from_vectors(a, b):
    """Rotation matrix that rotates unit vector a onto unit vector b (Rodrigues' formula)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, ortho)
        v = v / np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        return np.eye(3) + 2 * vx @ vx
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def detect_floor_rotation(pcd, distance_threshold, ransac_n, num_iterations):
    """[floor_plane method] RANSAC-fit the dominant plane and return (R, pivot)
    that rotates its normal onto +Z, about the inlier centroid."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    points = np.asarray(pcd.points)
    inlier_points = points[inliers]
    centroid = points.mean(axis=0)
    inlier_centroid = inlier_points.mean(axis=0)

    to_centroid = centroid - inlier_centroid
    if np.dot(normal, to_centroid) < 0:
        normal = -normal

    target = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_from_vectors(normal, target)
    print(f"[floor_plane] normal={normal}, {len(inliers)}/{len(points)} inlier points")
    return R, inlier_centroid


def local_axis_vector(axis_str):
    sign = -1.0 if axis_str.startswith("-") else 1.0
    axis = axis_str.lstrip("-")
    vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    return sign * np.array(vec, dtype=np.float64)


def estimate_up_rotation_from_trajectory(rotations, cam_up_axis):
    """[camera_up method] Average the world-frame 'up' direction of every camera
    pose to estimate gravity, then return (R, pivot=origin) that rotates it onto +Z."""
    up_local = local_axis_vector(cam_up_axis)
    up_world_per_frame = rotations @ up_local  # (N,3,3) @ (3,) -> (N,3)
    up_world_per_frame = up_world_per_frame / np.linalg.norm(up_world_per_frame, axis=1, keepdims=True)

    mean_up = up_world_per_frame.mean(axis=0)
    consistency = np.linalg.norm(mean_up)  # 1.0 = perfectly consistent, closer to 0 = camera rolled around a lot
    mean_up = mean_up / consistency

    print(f"[camera_up] estimated up vector = {mean_up}, consistency = {consistency:.3f} (1.0 = fully consistent)")
    if consistency < 0.5:
        print("[camera_up] WARNING: low consistency, the camera rolled/tumbled a lot during capture. "
              "This estimate may be unreliable, consider --align_method floor_plane instead, or check --cam_up_axis.")

    target = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_from_vectors(mean_up, target)
    return R, np.zeros(3)


def quat_to_rotmat(qx, qy, qz, qw):
    """Standard quaternion (x,y,z,w) to 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def load_trajectory(traj_file):
    """Load TUM-format poses.txt: frame_id tx ty tz qx qy qz qw.
    Returns positions (N,3) and rotation matrices (N,3,3), in file order."""
    positions = []
    rotations = []
    with open(traj_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = vals
            positions.append([tx, ty, tz])
            rotations.append(quat_to_rotmat(qx, qy, qz, qw))
    return np.array(positions), np.array(rotations)


def moving_average(points, window):
    if window <= 1:
        return points
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(points)
    for dim in range(points.shape[1]):
        padded = np.pad(points[:, dim], (window // 2, window // 2), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")[: points.shape[0]]
    return smoothed


def resample_by_arclength(positions, num_samples):
    """Resample a polyline to num_samples points, evenly spaced by arc length."""
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(deltas)])
    total = cumdist[-1]
    if total < 1e-6:
        return np.repeat(positions[:1], num_samples, axis=0), cumdist, total
    targets = np.linspace(0.0, total, num_samples)
    resampled = np.stack([
        np.interp(targets, cumdist, positions[:, i]) for i in range(positions.shape[1])
    ], axis=1)
    return resampled, cumdist, total


def point_at_arclength(positions, cumdist, s):
    """Get the interpolated point on the polyline at arc-length distance s (clamped)."""
    s = np.clip(s, cumdist[0], cumdist[-1])
    return np.stack([
        np.interp(s, cumdist, positions[:, i]) for i in range(positions.shape[1])
    ])


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it first, e.g. `sudo apt install ffmpeg`.")

    if not os.path.exists(args.pcd):
        raise FileNotFoundError(f"Point cloud file not found: {args.pcd}")

    if args.mode == "trajectory" and not args.traj_file:
        raise ValueError("--traj_file is required when --mode trajectory")
    if args.align_method == "camera_up" and not args.traj_file:
        raise ValueError("--traj_file is required when --align_method camera_up (needs the recorded poses to estimate gravity)")

    print(f"Loading point cloud from {args.pcd}...")
    pcd = o3d.io.read_point_cloud(args.pcd)
    if len(pcd.points) == 0:
        raise RuntimeError("Loaded point cloud has 0 points. Check the input file.")
    print(f"Loaded {len(pcd.points)} points")

    if args.remove_outliers:
        print("Removing statistical outliers...")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
        print(f"Remaining after outlier removal: {len(pcd.points)} points")

    # Load trajectory early if needed for alignment and/or flythrough mode.
    traj_positions, traj_rotations = None, None
    if args.traj_file:
        print(f"Loading trajectory from {args.traj_file}...")
        traj_positions, traj_rotations = load_trajectory(args.traj_file)
        if len(traj_positions) < 2:
            raise RuntimeError("Trajectory file has fewer than 2 poses.")
        print(f"Loaded {len(traj_positions)} recorded poses")

    # Alignment: compute rotation once, apply consistently to both the cloud and (if used) the trajectory.
    R_align = None
    align_pivot = None
    if args.align_method == "floor_plane":
        print("Aligning via RANSAC floor plane detection...")
        R_align, align_pivot = detect_floor_rotation(
            pcd,
            distance_threshold=args.floor_thresh,
            ransac_n=args.floor_ransac_n,
            num_iterations=args.floor_iterations,
        )
    elif args.align_method == "camera_up":
        print("Aligning via averaged camera-up direction from the trajectory...")
        R_align, align_pivot = estimate_up_rotation_from_trajectory(traj_rotations, args.cam_up_axis)

    if R_align is not None:
        pcd.rotate(R_align, center=align_pivot)

    if args.mode == "orbit" and args.crop_radius is not None:
        points = np.asarray(pcd.points)
        center = np.array(args.crop_center) if args.crop_center is not None else points.mean(axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        mask = dists <= args.crop_radius
        pcd = pcd.select_by_index(np.where(mask)[0])
        print(f"Cropped to radius {args.crop_radius} around {center}: {len(pcd.points)} points remaining")

    if len(pcd.points) == 0:
        raise RuntimeError("No points left after filtering/cropping. Loosen --crop_radius or outlier settings.")

    if args.voxel_size is not None:
        pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"Downsampled to {len(pcd.points)} points")

    # Center the cloud at the origin. Keep track of the shift so we can apply the
    # exact same rigid transform (rotate about align_pivot, then subtract this
    # center) to the trajectory in trajectory mode.
    cloud_center = pcd.get_center()
    pcd.translate(-cloud_center)

    frames_dir = args.keep_frames_dir or tempfile.mkdtemp(prefix="render_frames_")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Rendering frames to {frames_dir}...")

    renderer = rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background(args.bg_color)

    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = args.point_size

    renderer.scene.add_geometry("pcd", pcd, material)

    aspect = args.width / args.height
    frame_idx = 0

    def render_and_save(eye, lookat, up, far_hint):
        nonlocal frame_idx
        renderer.scene.camera.set_projection(
            args.fov_deg, aspect, 0.05, max(far_hint * 4.0, 20.0), rendering.Camera.FovType.Vertical
        )
        renderer.scene.camera.look_at(lookat, eye, up)
        img = renderer.render_to_image()
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        o3d.io.write_image(frame_path, img)
        frame_idx += 1

    if args.mode == "orbit":
        lookat = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        print(f"Phase 1: outer orbit at radius {args.start_radius} over {args.outer_orbit_frames} frames...")
        for i in range(args.outer_orbit_frames):
            azimuth = args.start_azimuth_deg + 360.0 * (i / args.outer_orbit_frames)
            eye = spherical_to_cartesian(args.start_radius, azimuth, args.elevation_deg) + lookat
            render_and_save(eye, lookat, up, args.start_radius)

        # End of outer orbit lands back at start_azimuth_deg (full 360 loop), so
        # the zoom-in phase continues from that same azimuth for a seamless cut.
        zoom_azimuth = args.start_azimuth_deg
        print(f"Phase 2: zooming from radius {args.start_radius} to {args.end_radius} over {args.zoom_frames} frames...")
        for i in range(args.zoom_frames):
            t = i / max(args.zoom_frames - 1, 1)
            radius = args.start_radius + (args.end_radius - args.start_radius) * t
            eye = spherical_to_cartesian(radius, zoom_azimuth, args.elevation_deg) + lookat
            render_and_save(eye, lookat, up, radius)

        print(f"Phase 3: inner orbit at radius {args.end_radius} over {args.inner_orbit_frames} frames...")
        for i in range(args.inner_orbit_frames):
            azimuth = zoom_azimuth + 360.0 * (i / args.inner_orbit_frames)
            eye = spherical_to_cartesian(args.end_radius, azimuth, args.elevation_deg) + lookat
            render_and_save(eye, lookat, up, args.end_radius)

    else:  # trajectory
        positions = traj_positions

        if R_align is not None:
            positions = (positions - align_pivot) @ R_align.T + align_pivot

        positions = positions - cloud_center

        positions_smooth = moving_average(positions, args.traj_smooth_window)

        num_frames = max(int(args.traj_duration * args.fps), 2)
        path, cumdist, total_len = resample_by_arclength(positions_smooth, num_frames)
        print(f"Path length: {total_len:.2f} m, rendering {num_frames} frames over {args.traj_duration:.1f}s")

        up = np.array(args.traj_up, dtype=np.float64)
        up = up / np.linalg.norm(up)

        for i in range(num_frames):
            eye = path[i].copy()
            eye[2] += args.height_offset

            target_s = min(cumdist[-1] * (i / max(num_frames - 1, 1)) + args.look_ahead_dist, cumdist[-1])
            lookat = point_at_arclength(path, np.linspace(0, total_len, num_frames), target_s)
            lookat[2] += args.height_offset

            if np.linalg.norm(lookat - eye) < 1e-4:
                lookat = eye + np.array([0.01, 0.0, 0.0])

            render_and_save(eye, lookat, up, max(total_len, 5.0))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Encoding video to {args.out}...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        args.out,
    ]
    subprocess.run(cmd, check=True)
    print(f"Done: {args.out}")

    if args.keep_frames_dir is None:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
