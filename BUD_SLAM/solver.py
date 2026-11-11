from collections import deque

import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d
from termcolor import colored
from scipy.linalg import rq
import glob
import os
import torch.nn.functional as F


from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from vggt_slam.slam_utils import compute_image_embeddings, Accumulator
from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.graph import PoseGraph
from vggt_slam.scale_solver import estimate_scale_pairwise
from vggt_slam.viewer import Viewer
import vggt_slam.slam_utils as utils

# --- Uncertainty estimation modules ---
from vggt_slam.uncertainty import compute_point_uncertainty, uncertainty_to_color
from vggt_slam.quantum_uncertainty import compute_latent_uncertainty

DEBUG = False

def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        vis_imgs: bool = False,
        dynamic_conf_threshold: float = None,
        enable_dynamic_split: bool = False,
        dynamic_residual_thresh: float = 0.08,
        dynamic_window_frames: int = 5):
        """
        Args:
            enable_dynamic_split: if True, use disagreement between VGGT's
                depth_head and point_head 3D estimates to flag likely-dynamic
                points. Flagged points are excluded from the persistent
                static map and instead shown via a separate sliding-window
                point cloud (see update_dynamic_window), avoiding the
                ghosting/trailing artifacts caused by merging moving-object
                points across frames into one static cloud.

                Off by default: relies on the assumption that depth_head and
                point_head outputs share the same per-submap coordinate
                frame, which has not been independently verified.
            dynamic_residual_thresh: relative disagreement threshold (fraction
                of point distance from the submap origin) above which a point
                is flagged dynamic.
            dynamic_window_frames: number of most recent frames' dynamic
                points kept visible at once in the sliding window.
        """

        self.init_conf_threshold = init_conf_threshold
        self.dynamic_conf_threshold = dynamic_conf_threshold
        self.vis_voxel_size = vis_voxel_size
        self.vis_imgs = vis_imgs

        # self.enable_dynamic_split = enable_dynamic_split
        # self.dynamic_residual_thresh = dynamic_residual_thresh
        # self.dynamic_window = deque(maxlen=dynamic_window_frames) if enable_dynamic_split else None
        
        self.enable_dynamic_split = enable_dynamic_split
        self.dynamic_residual_thresh = dynamic_residual_thresh
        self.dynamic_window_frames = dynamic_window_frames
        self.dynamic_frame_counter = 0

        # self.viewer = Viewer()
        # if self.enable_dynamic_split:
        #     self.viewer.setup_dynamic_playback_gui(window_frames=dynamic_window_frames)

        print(f"[solver] enable_dynamic_split = {self.enable_dynamic_split}")
        self.viewer = Viewer()
        if self.enable_dynamic_split:
            print("[solver] calling setup_dynamic_playback_gui")
            self.viewer.setup_dynamic_playback_gui(window_frames=dynamic_window_frames)
            print("[solver] setup_dynamic_playback_gui returned OK")


        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        # self.image_retrieval = None  # TODO: dino_salad.ckpt 下载太慢，临时禁用回环检测
        self.current_working_submap = None

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.vggt_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

        self.dyn_masks_path = "/home/gdy/VGGT-SLAM/dyn_mask/dyn_masks.pt"
        self.all_frame_names = sorted(glob.glob("/home/gdy/VGGT-SLAM/tesse_cd_apartment/*.jpg"))


    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if points_in_world_frame is None or len(points_in_world_frame) == 0:
            return
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap, static_voxels=None, voxel_size=None):
        if static_voxels is not None:
            points_in_world_frame, points_colors = submap.get_points_and_colors_in_world_frame(self.graph, static_voxels=static_voxels, voxel_size=voxel_size)
        else:
            points_in_world_frame = submap.get_points_in_world_frame(self.graph)
            points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames() if self.vis_imgs else None
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self, static_voxels=None, voxel_size=None):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap, static_voxels=static_voxels, voxel_size=voxel_size)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def set_submap_uncertainty_vis(self, submap_id, points_local, uncertainty, name_suffix="uncertainty"):
        """
        Render a color-coded uncertainty heatmap for one submap as a
        separate, independently toggleable point cloud layer in the
        viewer (named "<submap_id>_<name_suffix>", distinct from the
        submap's normal colored cloud).

        Transforms points_local using a single per-submap homography
        (graph.get_homography(submap_id), the node added for frame 0 by
        add_edge), not a per-frame transform: all frames within one
        submap already share one local coordinate frame from VGGT, the
        same assumption add_edge's scale estimation relies on when it
        indexes get_frame_pointcloud directly without a per-frame
        transform. Must be called after add_edge has registered this
        submap's node in the graph.

        Args:
            submap_id: id of the submap these points belong to.
            points_local: (S, H, W, 3) in the submap's local frame, same
                array passed to add_all_points.
            uncertainty: (S, H, W) in [0, 1], same leading shape as
                points_local.
            name_suffix: suffix used to name this layer, so multiple
                uncertainty sources (e.g. point-based vs latent-based)
                can coexist as separate toggleable layers.
        """
        H0 = self.graph.get_homography(submap_id)
        pts = points_local.reshape(-1, 3)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_h = (H0 @ pts_h.T).T
        pts_world = (pts_h[:, :3] / pts_h[:, 3:]).astype(np.float32)

        colors = uncertainty_to_color(uncertainty.reshape(-1))
        name = f"{submap_id}_{name_suffix}"
        self.set_point_cloud(pts_world, colors, name, 0.001)

    def update_dynamic_window(self):
        """
        Push each frame's dynamic points from the just-processed submap to the
        viewer as an individual point cloud handle (keyed by a global,
        ever-increasing frame index), then update which frames are visible
        based on the live sliding window. Old frames are kept, not deleted,
        so the Dynamic Playback GUI can scrub back through full history.

        No-op if enable_dynamic_split was not set on this Solver.
        """
        if not self.enable_dynamic_split:
            return

        submap = self.current_working_submap
        if submap is None:
            return

        for frame_index in range(submap.get_num_frames()):
            pts, cols = submap.get_dynamic_points_and_colors_frame(frame_index, self.graph)
            global_idx = self.dynamic_frame_counter
            self.dynamic_frame_counter += 1
            if pts is None or len(pts) == 0:
                continue
            self.viewer.add_dynamic_frame_points(pts, cols, global_idx)

        self.viewer.set_live_dynamic_window(self.dynamic_frame_counter - 1, self.dynamic_window_frames)


    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T # TODO Dominic check if we want to use P_prior here
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)

        return proj_mat_world_to_cam, world_points

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False):
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        if submap_id_prev is not None:
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Estimate scale factor between submaps.
            prior_submap = self.map.get_submap(submap_id_prev)

            # current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            # prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            # good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())
            # good_mask = good_mask.reshape(-1)

            # if np.sum(good_mask) < 100:
            #     print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
            #     good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
            #     if np.sum(good_mask) < 100: # Handle the case where loop closure frames do not have enough points.
            #         good_mask = (prior_conf > 0).reshape(-1)

            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)

            # Exclude points flagged as dynamic (either side) from scale estimation.
            # get_dynamic_mask_frame returns None when enable_dynamic_split wasn't set
            # for that submap, in which case this is a no-op (all True).
            current_dyn = current_submap.get_dynamic_mask_frame(frame_id_curr)
            prior_dyn = prior_submap.get_dynamic_mask_frame(frame_id_prev)
            not_dynamic = np.ones_like(prior_conf, dtype=bool)
            if current_dyn is not None:
                not_dynamic &= ~current_dyn
            if prior_dyn is not None:
                not_dynamic &= ~prior_dyn

            good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold()) * not_dynamic
            good_mask = good_mask.reshape(-1)

            if np.sum(good_mask) < 100:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = ((prior_conf > prior_submap.get_conf_threshold()) * not_dynamic).reshape(-1)
                if np.sum(good_mask) < 100:
                    # Last resort (e.g. loop closure frames): drop dynamic filter too,
                    # matches original behavior for this degraded case.
                    good_mask = (prior_conf > 0).reshape(-1)


            P_temp = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
            t1 = (P_temp[0:3,0:3] @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            print(colored("scale factor", 'green'), scale_factor_est_output)
            scale_factor = scale_factor_est_output[0]
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            H_overlap_prior_overlap_current = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0] @ H_scale
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for intra submaps constraint.
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, self.graph.intra_submap_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        world_to_cam = current_submap.get_all_poses()
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose) # TODO Dominic, no need to take the inverse twice, just use cam_to_world
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Add between factor for inner submap constraint.
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
                "dyn_masks": (S, H, W) or None  - 语义动态mask, 1表示动态
                "latent_uncertainty": (S, H, W) or None  - quantum-inspired
                    latent-token uncertainty from compute_latent_uncertainty
            }
        """
        # Unpack prediction dict
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # Independent 3D estimate from point_head, used only for dynamic-point
        # detection (disagreement with the depth_head-based world_points above).
        world_points_alt = pred_dict.get("world_points") if self.enable_dynamic_split else None

        # Semantic dynamic mask (e.g. person detection), highest-priority signal
        # for excluding points from the static map, independent of geometric heuristics.
        dyn_mask = pred_dict.get("dyn_masks")  # (S, H, W), 1 = dynamic, may be None

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]

        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)

        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        first_edge = submap_id_prev is None

        if not first_edge:
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(
            world_points, colors, conf, self.init_conf_threshold, K_4x4,
            dynamic_filter_percentile=self.dynamic_conf_threshold,
            points_alt=world_points_alt,
            dynamic_residual_thresh=self.dynamic_residual_thresh if self.enable_dynamic_split else None,
            dyn_mask=dyn_mask,
        )
        self.current_working_submap.set_conf_masks(conf)

        # --- Uncertainty estimation ---
        # 1) Per-point uncertainty: combines model confidence (depth_conf)
        #    with depth_head/point_head disagreement (when available).
        #    Attached directly to the submap object rather than routed
        #    through add_all_points, since Submap's constructor/storage
        #    for this field is not something this module owns.
        point_uncertainty = compute_point_uncertainty(conf, world_points, world_points_alt)
        self.current_working_submap.point_uncertainty = point_uncertainty

        # 2) Latent-token uncertainty: quantum-inspired (density matrix /
        #    von Neumann entropy) uncertainty computed upstream in
        #    run_predictions from the aggregator's latent tokens. Attached
        #    here for the same reason as point_uncertainty above.
        latent_uncertainty = pred_dict.get("latent_uncertainty")
        if latent_uncertainty is not None:
            self.current_working_submap.latent_uncertainty = latent_uncertainty

        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False)

        # Render uncertainty heatmaps as separate toggleable layers. Must
        # happen after add_edge, since that call is what registers this
        # submap's homography node in the graph.
        self.set_submap_uncertainty_vis(submap_id_curr, world_points, point_uncertainty)
        if latent_uncertainty is not None:
            self.set_submap_uncertainty_vis(submap_id_curr, world_points, latent_uncertainty,
                                             name_suffix="latent_uncertainty")

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"])
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1)
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W)

            intrinsics_cam = pred_dict["intrinsic_lc"]

            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)
            world_points_lc_alt = pred_dict.get("world_points_lc") if self.enable_dynamic_split else None

            # Semantic dynamic mask for the loop closure frame pair, if available.
            dyn_mask_lc = pred_dict.get("dyn_masks_lc")  # (2, H, W), may be None

            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(
                world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc,
                dynamic_filter_percentile=self.dynamic_conf_threshold,
                points_alt=world_points_lc_alt,
                dynamic_residual_thresh=self.dynamic_residual_thresh if self.enable_dynamic_split else None,
                dyn_mask=dyn_mask_lc,
            )
            print("Loop closure conf", conf_lc.shape)
            print(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame)
            lc_submap.set_conf_masks(conf_lc)

            # Per-point uncertainty for the loop closure submap, same
            # signal as the main submap above.
            lc_point_uncertainty = compute_point_uncertainty(conf_lc, world_points_lc, world_points_lc_alt)
            lc_submap.point_uncertainty = lc_point_uncertainty

            self.map.add_submap(lc_submap)

            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True)

            # Loop closure submap's node is registered by the first
            # add_edge call above (is_loop_closure=False branch), so the
            # heatmap can be rendered right after.
            self.set_submap_uncertainty_vis(lc_submap_num, world_points_lc, lc_point_uncertainty)


    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords
    
    def _load_dyn_masks_for_frames(self, image_names, target_hw, device, dtype):
        """
        根据 image_names 从预先计算好的全量 dyn_masks 中切出对应帧，
        并 resize 到当前 images 的分辨率。

        self.dyn_masks_path: 预先跑好的 dyn_masks.pt 路径
        self.all_frame_names: 生成 dyn_masks.pt 时用的全量帧文件名列表 (顺序必须一致)
        """
        if self.dyn_masks_path is None:
            return None

        if not hasattr(self, "_full_dyn_masks"):
            self._full_dyn_masks = torch.load(self.dyn_masks_path, map_location="cpu")  # [S_total, H, W]

        name_to_idx = {os.path.basename(n): i for i, n in enumerate(self.all_frame_names)}
        print("DEBUG: Dict keys sample ->", list(name_to_idx.keys())[:3])
        indices = [name_to_idx[os.path.basename(n)] for n in image_names]

        dyn_masks_submap = self._full_dyn_masks[indices]  # [S_submap, H, W]

        if dyn_masks_submap.shape[-2:] != target_hw:
            dyn_masks_submap = F.interpolate(
                dyn_masks_submap.unsqueeze(1), size=target_hw, mode="nearest"
            ).squeeze(1)

        return dyn_masks_submap.to(device).to(dtype)
    
    def _load_dyn_mask_for_retrieved_frame(self, loop_info, target_hw, device, dtype):
        """
        为loop closure检索到的历史帧找到对应的dyn_mask。
        """
        if getattr(self, "dyn_masks_path", None) is None or not hasattr(self, "_full_dyn_masks"):
            return None

        retrieved_submap = self.map.get_submap(loop_info.detected_submap_id)
        retrieved_name = retrieved_submap.get_img_names_at_index(loop_info.detected_submap_frame)

        name_to_idx = {os.path.basename(n): i for i, n in enumerate(self.all_frame_names)}
        if os.path.basename(retrieved_name) not in name_to_idx:
            return None

        idx = name_to_idx[os.path.basename(retrieved_name)]
        mask = self._full_dyn_masks[idx:idx + 1]  # [1, H, W]

        if mask.shape[-2:] != target_hw:
            mask = F.interpolate(mask.unsqueeze(1), size=target_hw, mode="nearest").squeeze(1)

        return mask.to(device).to(dtype)



    def run_predictions(self, image_names, model, max_loops, clip_model, clip_preprocess):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t1 = time.time()
        with self.vggt_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(f"Loaded and preprocessed {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # --- 加载并对齐当前submap对应的dyn_masks ---
        dyn_masks = self._load_dyn_masks_for_frames(image_names, images.shape[-2:], device, dtype)

        # First submap so set new pcd num to 0
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        with torch.no_grad():
            t1 = time.time()
            with self.vggt_timer:
                predictions, qk_dict, enc_feat, aggregated_tokens_list = model(
                    images.to(dtype), dyn_masks=dyn_masks
                )  # for submap
            print(f"VGGT model inference took {time.time() - t1:.2f} seconds")

        # --- Latent-token uncertainty (quantum density-matrix / von Neumann
        # entropy formalism, see vggt_slam/quantum_uncertainty.py) ---
        # NOTE: relies on the assumed layout of aggregated_tokens_list
        # (prefix tokens then row-major patch tokens at model.aggregator's
        # target_layer). This has not been independently verified against
        # the model source; compute_latent_uncertainty raises a clear
        # shape-mismatch error rather than silently producing wrong output
        # if the assumption is off.
        target_layer = model.aggregator.target_layer
        layer_tokens = aggregated_tokens_list[target_layer][0]  # drop batch dim
        latent_uncertainty = compute_latent_uncertainty(
            layer_tokens,
            patch_start_idx=model.aggregator.patch_start_idx,
            image_height=images.shape[-2],
            image_width=images.shape[-1],
            patch_size=model.aggregator.patch_size,
        )

        # Check for loop closures and add retrieval vectors from new submap to the database
        predictions_lc = None  # lc stands for loop-closure
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame), retrieved_frames[0]), axis=0)

                # --- 为loop closure的两帧准备对应的dyn_masks ---
                lc_dyn_masks = None
                if dyn_masks is not None:
                    query_frame_idx = detected_loops[0].query_submap_frame
                    query_mask = dyn_masks[query_frame_idx:query_frame_idx + 1]  # [1, H, W]
                    retrieved_mask = self._load_dyn_mask_for_retrieved_frame(
                        detected_loops[0], images.shape[-2:], device, dtype
                    )  # [1, H, W]
                    if retrieved_mask is not None:
                        lc_dyn_masks = torch.cat([query_mask, retrieved_mask], dim=0)  # [2, H, W]

                predictions_lc, qk_dict_lc, enc_feat_lc, _ = model(lc_frames.to(dtype), dyn_masks=lc_dyn_masks)
                lc_H, lc_W = lc_frames.shape[-2:]
                predictions_lc["image_match_ratio"] = utils.compute_loop_closure_match_ratio(
                    qk_dict_lc,
                    target_layer=model.aggregator.target_layer,
                    patch_start_idx=model.aggregator.patch_start_idx,
                    image_height=lc_H // model.aggregator.patch_size,
                    image_width=lc_W // model.aggregator.patch_size,
                )

                loop_closure_frame_names = [new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame),
                self.map.get_submap(detected_loops[0].detected_submap_id).get_img_names_at_index(detected_loops[0].detected_submap_frame)]

            # Visualize loop closure frames
            if DEBUG:
                imgs = lc_frames.permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions["detected_loops"] = detected_loops

        if predictions_lc is not None:
            image_match_ratio = predictions_lc["image_match_ratio"]
            if image_match_ratio < 0.95:
                print(colored("Loop closure image match ratio too low, skipping loop closure", "red"))
                predictions_lc = None  # We set to None to ignore the loop closure
                predictions["detected_loops"] = []
            else:
                self.graph.increment_loop_closure()
                extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], retrieved_frames[0].shape[-2:])
                predictions["extrinsic_lc"] = extrinsic_lc
                predictions["intrinsic_lc"] = intrinsic_lc
                predictions["depth_lc"] = predictions_lc["depth"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"]
                predictions["world_points_lc"] = predictions_lc["world_points"]
                predictions["dyn_masks_lc"] = lc_dyn_masks.float().cpu().numpy() if lc_dyn_masks is not None else None


        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor) and key != "target_tokens":
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames[0:2,...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names
        
        predictions["dyn_masks"] = dyn_masks.float().cpu().numpy() if dyn_masks is not None else None
        predictions["latent_uncertainty"] = latent_uncertainty.float().cpu().numpy()

        return predictions
