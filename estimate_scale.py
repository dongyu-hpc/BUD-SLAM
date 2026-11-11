"""
用GT点云直接估算尺度校正系数，不需要传感器标定/双目/深度数据。

流程:
1. 检测pred/gt点云在各轴上的extent排序，判断坐标轴是否错位对应
2. 如果错位，先做轴置换，让两者的"层高/宽/长"这三个维度对应上
3. 用置换后的extent比例估算尺度
4. 缩放pred点云后，做ICP精修(旋转平移)
"""

import numpy as np
import open3d as o3d


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
    """
    通过比较extent的排序(小->大), 找到把pred的轴重排到与gt对应的置换。
    返回一个长度为3的index数组, 使得 pred_pts[:, perm] 的轴顺序与gt对齐。
    """
    pred_extent = robust_extent(pred_pts)
    gt_extent = robust_extent(gt_pts)

    pred_order = np.argsort(pred_extent)  # 按extent从小到大排序后的轴index
    gt_order = np.argsort(gt_extent)

    print(f"pred extent: {pred_extent}, order(small->large): {pred_order}")
    print(f"gt extent: {gt_extent}, order(small->large): {gt_order}")

    # perm[i] = 应该放到位置i的pred轴序号
    # 使得 pred_pts[:, perm] 的axis order 与 gt_order 一致
    perm = np.empty(3, dtype=int)
    for rank in range(3):
        pred_axis = pred_order[rank]
        gt_axis = gt_order[rank]
        perm[gt_axis] = pred_axis

    print(f"axis permutation (pred -> gt-aligned): {perm}")
    return perm


def estimate_scale_from_extent(pred_pts, gt_pts, robust_percentile=(2, 98)):
    pred_extent = robust_extent(pred_pts, robust_percentile)
    gt_extent = robust_extent(gt_pts, robust_percentile)

    ratios = gt_extent / pred_extent
    print(f"per-axis scale ratios (after alignment): {ratios}")

    scale = np.median(ratios)
    print(f"estimated scale factor: {scale:.4f}")
    return scale


def align_with_scale(pred_pts, gt_pts, scale, voxel_size=0.05, threshold=0.1):
    pred_centroid = pred_pts.mean(axis=0)
    pred_scaled = (pred_pts - pred_centroid) * scale + pred_centroid

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_scaled)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)

    pred_pcd = pred_pcd.voxel_down_sample(voxel_size)
    gt_pcd = gt_pcd.voxel_down_sample(voxel_size)

    init_translation = gt_pcd.get_center() - pred_pcd.get_center()
    init_transform = np.eye(4)
    init_transform[:3, 3] = init_translation

    reg = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, max_correspondence_distance=threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"ICP fitness={reg.fitness:.4f}, inlier_rmse={reg.inlier_rmse:.4f}")
    return reg.transformation, pred_scaled


if __name__ == "__main__":
    pred_pts = load_points("outputs/apartment_points.pcd")
    gt_pts = load_points("tesse_cd_apartment/apartment_gt/gt_background.ply")

    print("=== Step 1: detect axis permutation ===")
    perm = find_axis_permutation(pred_pts, gt_pts)
    pred_pts_reordered = pred_pts[:, perm]

    print("\n=== Step 2: estimate scale after axis alignment ===")
    scale = estimate_scale_from_extent(pred_pts_reordered, gt_pts)

    print("\n=== Step 3: ICP refine (rotation + translation only) ===")
    # transform, pred_scaled = align_with_scale(pred_pts_reordered, gt_pts, scale)
    transform, pred_scaled = align_with_scale(pred_pts_reordered, gt_pts, scale, threshold=0.05)


    pred_final = (transform[:3, :3] @ pred_scaled.T).T + transform[:3, 3]

    np.save("pred_points_scale_corrected.npy", pred_final)
    print(f"\nFinal scale factor: {scale:.4f}")
    print(f"Axis permutation applied: {perm}")
    print("Saved corrected points to pred_points_scale_corrected.npy")
