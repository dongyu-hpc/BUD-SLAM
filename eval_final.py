"""
完整几何评测: 轴对齐 + 尺度校正 + ICP配准 + Chamfer/F-score计算

(vggt-slam) gdy@DESKTOP-RS1KQEU:~/VGGT-SLAM$ python eval_final.py
[Open3D WARNING] geometry::TriangleMesh appears to be a geometry::PointCloud (only contains vertices, but no triangles).
=== Step 1: axis alignment ===
axis permutation: [2 0 1]

=== Step 2: scale estimation ===
scale factor: 7.8774

=== Step 3: ICP (rotation + translation) ===
ICP fitness=0.5238, inlier_rmse=0.1608

=== Step 4: Chamfer / F-score ===
  chamfer_distance: 0.448805
  accuracy_mean: 0.331604
  completeness_mean: 0.117201
  precision: 0.240522
  recall: 0.686366
  fscore: 0.356216
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_points(path):
    if path.endswith(".ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) > 0:
            return np.asarray(mesh.vertices)
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def robust_extent(pts, robust_percentile=(2, 98)):
    lo, hi = robust_percentile
    return np.percentile(pts, hi, axis=0) - np.percentile(pts, lo, axis=0)


def find_axis_permutation(pred_pts, gt_pts):
    pred_extent = robust_extent(pred_pts)
    gt_extent = robust_extent(gt_pts)
    pred_order = np.argsort(pred_extent)
    gt_order = np.argsort(gt_extent)

    perm = np.empty(3, dtype=int)
    for rank in range(3):
        perm[gt_order[rank]] = pred_order[rank]
    return perm


def estimate_scale(pred_pts, gt_pts):
    pred_extent = robust_extent(pred_pts)
    gt_extent = robust_extent(gt_pts)
    return np.median(gt_extent / pred_extent)


def icp_align(pred_pts, gt_pts, voxel_size=0.05, threshold=0.3):
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_pts)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)

    pred_pcd = pred_pcd.voxel_down_sample(voxel_size)
    gt_pcd = gt_pcd.voxel_down_sample(voxel_size)

    init_transform = np.eye(4)
    init_transform[:3, 3] = gt_pcd.get_center() - pred_pcd.get_center()

    reg = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, max_correspondence_distance=threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"ICP fitness={reg.fitness:.4f}, inlier_rmse={reg.inlier_rmse:.4f}")
    return reg.transformation


def compute_chamfer_and_fscore(pred_pts, gt_pts, threshold=0.1, voxel_size=0.02):
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_pts)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)

    pred_pcd = pred_pcd.voxel_down_sample(voxel_size)
    gt_pcd = gt_pcd.voxel_down_sample(voxel_size)

    pred_ds = np.asarray(pred_pcd.points)
    gt_ds = np.asarray(gt_pcd.points)

    gt_tree = cKDTree(gt_ds)
    pred_tree = cKDTree(pred_ds)

    dist_pred_to_gt, _ = gt_tree.query(pred_ds, k=1, workers=-1)
    dist_gt_to_pred, _ = pred_tree.query(gt_ds, k=1, workers=-1)

    chamfer_dist = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
    precision = (dist_pred_to_gt < threshold).mean()
    recall = (dist_gt_to_pred < threshold).mean()
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "chamfer_distance": chamfer_dist,
        "accuracy_mean": dist_pred_to_gt.mean(),
        "completeness_mean": dist_gt_to_pred.mean(),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


if __name__ == "__main__":
    pred_pts = load_points("outputs/apartment_points.pcd")
    gt_pts = load_points("tesse_cd_apartment/apartment_gt/gt_background.ply")

    print("=== Step 1: axis alignment ===")
    perm = find_axis_permutation(pred_pts, gt_pts)
    pred_pts = pred_pts[:, perm]
    print(f"axis permutation: {perm}")

    print("\n=== Step 2: scale estimation ===")
    scale = estimate_scale(pred_pts, gt_pts)
    pred_centroid = pred_pts.mean(axis=0)
    pred_pts = (pred_pts - pred_centroid) * scale + pred_centroid
    print(f"scale factor: {scale:.4f}")

    print("\n=== Step 3: ICP (rotation + translation) ===")
    transform = icp_align(pred_pts, gt_pts, threshold=0.3)
    pred_pts = (transform[:3, :3] @ pred_pts.T).T + transform[:3, 3]

    print("\n=== Step 4: Chamfer / F-score ===")
    results = compute_chamfer_and_fscore(pred_pts, gt_pts, threshold=0.1)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
