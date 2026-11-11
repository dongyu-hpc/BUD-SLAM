import re
import os
import cv2
import torch
import numpy as np
import scipy
import open3d as o3d
from vggt_slam.slam_utils import decompose_camera

class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.R_world_map = None
        self.poses = None
        self.frames = None
        self.proj_mats = None
        self.retrieval_vectors = None
        self.colors = None # (S, H, W, 3)
        self.conf = None # (S, H, W)
        self.conf_masks = None # (S, H, W)
        self.conf_threshold = None
        self.dynamic_conf_threshold = None
        self.pointclouds = None # (S, H, W, 3)
        self.voxelized_points = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None
        self.is_lc_submap = False
        self.img_names = []
        self.semantic_vectors = []
        # (S, H, W) bool mask, True = flagged as a dynamic (moving-object) point.
        # Only populated when Solver is run with enable_dynamic_split=True.
        self.dynamic_mask = None

    def set_lc_status(self, is_lc_submap):
        self.is_lc_submap = is_lc_submap

    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics_inv,
                    dynamic_filter_percentile=None, points_alt=None, dynamic_residual_thresh=None,
                    dyn_mask=None):
        """
        Args:
            points: (S, H, W, 3) world points, typically from depth_head output
                unprojected via camera intrinsics/extrinsics.
            colors: (S, H, W, 3)
            conf: (S, H, W) confidence from depth_head.
            conf_threshold_percentile: percentile of conf used as the base
                filtering threshold (points below this are dropped as noise).
            intrinsics_inv: (S, 4, 4) camera projection matrices for this submap.
            dynamic_filter_percentile: optional stricter percentile used as a
                secondary heuristic (low-confidence points are often dynamic
                objects with less consistent multi-view geometry).
            points_alt: optional (S, H, W, 3) independent estimate of the same
                3D points, e.g. VGGT's point_head direct regression output.
                If provided together with dynamic_residual_thresh, per-pixel
                disagreement between `points` and `points_alt` is used to flag
                dynamic points, which are then excluded from the persistent/
                static map (see filter_data_by_confidence, get_points_in_world_frame).

                ASSUMPTION (unverified): `points` and `points_alt` are expressed
                in the same coordinate frame (this submap's local world frame,
                anchored at the first camera). If this assumption doesn't hold,
                the residual will be dominated by a constant offset/rotation
                rather than true object motion, and this filter will misbehave
                (e.g. flagging everything or nothing as dynamic). Check the
                resulting dynamic_mask.mean() before trusting this feature.
            dynamic_residual_thresh: relative threshold (fraction of point
                distance from the submap origin) above which a point is
                flagged dynamic, e.g. 0.08 = 8% relative disagreement.
            dyn_mask: optional (S, H, W) semantic dynamic mask (e.g. from a
                person detector/segmenter), 1 = dynamic, 0 = static. This is
                treated as the most direct/trusted signal and is OR-combined
                with the geometric residual-based dynamic_mask (if any): a
                point is excluded if EITHER signal flags it as dynamic.
        """
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        self.conf_threshold = np.percentile(self.conf, conf_threshold_percentile) + 1e-6
        if dynamic_filter_percentile is not None:
            self.dynamic_conf_threshold = np.percentile(self.conf, dynamic_filter_percentile) + 1e-6
        self.proj_mats = intrinsics_inv

        self.dynamic_mask = None
        if points_alt is not None and dynamic_residual_thresh is not None:
            residual = np.linalg.norm(points - points_alt, axis=-1)  # (S, H, W)
            scale = np.linalg.norm(points, axis=-1) + 1e-6
            relative_residual = residual / scale
            self.dynamic_mask = relative_residual > dynamic_residual_thresh

        if dyn_mask is not None:
            dyn_mask_bool = dyn_mask.astype(bool) if not dyn_mask.dtype == bool else dyn_mask
            if self.dynamic_mask is not None:
                # OR-combine: a point is dynamic if either the semantic mask
                # or the geometric residual heuristic flags it.
                self.dynamic_mask = self.dynamic_mask | dyn_mask_bool
            else:
                self.dynamic_mask = dyn_mask_bool


    def set_img_names(self, img_names):
        self.img_names = img_names

    def add_all_frames(self, frames):
        self.frames = frames

    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def get_lc_status(self):
        return self.is_lc_submap

    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold

    def get_effective_conf_threshold(self):
        """
        Confidence threshold used when building/visualizing the map point cloud.

        If dynamic point filtering mode is enabled (dynamic_conf_threshold set),
        returns the stricter of the base and dynamic thresholds, so points with
        high depth uncertainty (often caused by dynamic objects) are excluded.
        Otherwise falls back to the base conf_threshold.
        """
        if self.dynamic_conf_threshold is not None:
            return max(self.conf_threshold, self.dynamic_conf_threshold)
        return self.conf_threshold

    def get_conf_masks_frame(self, index):
        return self.conf_masks[index]

    def get_frame_at_index(self, index):
        return self.frames[index, ...]

    def get_num_frames(self):
        return len(self.poses) if self.poses is not None else 0

    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    def get_img_names_at_index(self, index):
        return self.img_names[index]

    def get_all_frames(self):
        return self.frames

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_first_homography_world(self, graph):
        homography =  graph.get_homography(self.get_id())
        homography = homography / homography[-1,-1]
        return homography

    def get_last_homography_world(self, graph):
        """
        Get the last camera projection matrix of the submap that is not a
        loop closure frame.
        Returns a 4x4 matrix normalized so that the last element is 1.
        """
        homography = graph.get_homography(self.get_id() + self.get_last_non_loop_frame_index())
        homography = homography / homography[-1,-1]
        return homography

    def get_first_pose_world(self, graph):
        """
        Get the first camera projection matrix of the submap.
        Returns a 4x4 matrix normalized so that the last element is 1.
        """
        return np.linalg.inv(self.get_first_homography_world(graph))

    def get_last_pose_world(self, graph):
        """
        Get the last camera projection matrix of the submap that is not a
        loop closure frame.
        Returns a 4x4 matrix normalized so that the last element is 1.
        """
        projection_mat = graph.get_projection_matrix(self.get_id() + self.get_last_non_loop_frame_index())
        projection_mat = projection_mat / projection_mat[-1,-1]
        return projection_mat

    def get_all_poses(self):
        return self.poses

    def get_all_poses_world(self, graph, give_camera_mat=False):
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        poses = []
        for index, homography_world in enumerate(homography_list):
            projection_mat = self.proj_mats[index] @ np.linalg.inv(homography_world) # TODO HERE
            projection_mat = projection_mat / projection_mat[-1,-1]
            if give_camera_mat:
                poses.append(projection_mat)
            else:
                cal, rot, trans, scale = decompose_camera(projection_mat[0:3,:])

                pose = np.eye(4)
                pose[0:3, 0:3] = rot
                pose[0:3,3] = trans
                poses.append(pose)
        return np.stack(poses, axis=0)

    def get_frame_pointcloud(self, pose_index):
        return self.pointclouds[pose_index]

    def set_frame_ids(self, file_paths):
        """
        Extract the frame number (integer or decimal) from the file names,
        removing any leading zeros, and add them all to a list.

        Note: This does not include any of the loop closure frames.
        """
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks

    def set_all_semantic_vectors(self, semantic_vectors):
        self.semantic_vectors = semantic_vectors

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])

    def get_frame_ids(self):
        # Note this does not include any of the loop closure frames
        return self.frame_ids

    def filter_data_by_confidence(self, data):
        init_conf_mask = self.conf > self.get_effective_conf_threshold()
        if self.dynamic_mask is not None:
            init_conf_mask = init_conf_mask & (~self.dynamic_mask)
        return data[init_conf_mask]

    def get_points_list_in_world_frame(self, graph, rectifing_homographies=None):
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        point_list = []
        frame_id_list = []
        frame_conf_mask = []
        threshold = self.get_effective_conf_threshold()
        for index in  range(len(self.pointclouds)):
            points = self.pointclouds[index]
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            point_list.append(points_transformed)
            frame_id_list.append(self.frame_ids[index])
            conf_mask = self.conf_masks[index] > threshold
            frame_conf_mask.append(conf_mask)
        return point_list, frame_id_list, frame_conf_mask


    def get_points_in_world_frame(self, graph):
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        points_all = None
        threshold = self.get_effective_conf_threshold()
        for index in  range(len(self.pointclouds)):
            points = self.pointclouds[index]
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            conf_mask = self.conf_masks[index] > threshold
            if self.dynamic_mask is not None:
                conf_mask = conf_mask & (~self.dynamic_mask[index])

            points_transformed = points_transformed[conf_mask]
            if index == 0:
                points_all = points_transformed
            else:
                points_all = np.vstack([points_all, points_transformed])

        return points_all


    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                points = self.filter_data_by_confidence(self.pointclouds)
                points_flat = points.reshape(-1, 3)
                colors = self.filter_data_by_confidence(self.colors)
                colors_flat = colors.reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                if (nb_points > 0):
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(nb_points=nb_points,
                                                                                           radius=voxel_size * factor_for_outlier_rejection)
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        H_world_map = self.poses[0]
        points_transformed = (H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame

    def get_points_colors(self):
        colors = self.filter_data_by_confidence(self.colors)
        return colors.reshape(-1, 3)

    def get_all_semantic_vectors(self):
        return self.semantic_vectors


    def get_points_in_mask(self, frame_index, mask, graph):
        points = self.get_points_list_in_world_frame(graph)[0][frame_index]
        points_flat = points.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        points_in_mask = points_flat[mask_flat]
        return points_in_mask

    def get_points_and_colors_in_world_frame(self, graph, static_voxels=None, voxel_size=None):
        """
        Points+colors in world frame after confidence filtering (and dynamic
        filtering, if enabled), optionally also filtered by a static-voxel
        set (see GraphMap.compute_static_voxel_set) to drop likely-dynamic
        points detected via cross-submap temporal analysis.
        """
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        threshold = self.get_effective_conf_threshold()
        points_all = None
        colors_all = None
        for index in range(len(self.pointclouds)):
            points = self.pointclouds[index]
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            conf_mask = self.conf_masks[index] > threshold
            if self.dynamic_mask is not None:
                conf_mask = conf_mask & (~self.dynamic_mask[index])
            pts = points_transformed[conf_mask]
            cols = self.colors[index][conf_mask]

            if static_voxels is not None:
                voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
                keep = np.fromiter((tuple(v) in static_voxels for v in voxel_idx), dtype=bool, count=len(voxel_idx))
                pts = pts[keep]
                cols = cols[keep]

            if points_all is None:
                points_all = pts
                colors_all = cols
            else:
                points_all = np.vstack([points_all, pts])
                colors_all = np.vstack([colors_all, cols])

        return points_all, colors_all

    def get_dynamic_points_and_colors_frame(self, frame_index, graph):
        """
        Points flagged as dynamic for a single frame, transformed to world
        frame. Returns (None, None) if dynamic splitting is disabled for this
        submap. Used to feed a sliding-window dynamic point cloud instead of
        merging moving-object points permanently into the static map (which
        causes ghosting/trailing artifacts across frames).
        """
        if self.dynamic_mask is None:
            return None, None

        homography = graph.get_homography(frame_index + self.get_id())
        points = self.pointclouds[frame_index]
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (homography @ points_homogeneous.T).T
        points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

        threshold = self.get_effective_conf_threshold()
        conf_mask = self.conf_masks[frame_index] > threshold
        dyn_mask = self.dynamic_mask[frame_index] & conf_mask

        pts = points_transformed[dyn_mask]
        cols = self.colors[frame_index][dyn_mask]
        return pts, cols
    
    def get_dynamic_mask_frame(self, index):
        """
        Returns the (H, W) boolean dynamic mask for a single frame (True = flagged
        as dynamic), or None if dynamic splitting was not enabled for this submap.
        """
        if self.dynamic_mask is None:
            return None
        return self.dynamic_mask[index]