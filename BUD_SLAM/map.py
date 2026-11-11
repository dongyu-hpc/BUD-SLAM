import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.slam_utils import decompose_camera, cosine_similarity
from collections import defaultdict

# class GraphMap:
#     def __init__(self):
#         self.submaps = dict()
#         self.rectifying_H_mats = []
#         self.non_lc_submap_ids = []
    
#     def get_num_submaps(self):
#         return len(self.submaps)

#     def add_submap(self, submap):
#         submap_id = submap.get_id()
#         self.submaps[submap_id] = submap
#         if not submap.get_lc_status():
#             self.non_lc_submap_ids.append(submap_id)

#     def compute_static_voxel_set(self, graph, voxel_size=0.1, min_temporal_coverage=0.1, min_density=0.5):
#         voxel_to_frames = {}  # v -> set of frame_ids observed
#         global_min_frame = float("inf")
#         global_max_frame = float("-inf")

#         for submap in self.ordered_submaps_by_key():
#             point_list, frame_id_list, frame_conf_mask = submap.get_points_list_in_world_frame(graph)
#             for points, frame_id, conf_mask in zip(point_list, frame_id_list, frame_conf_mask):
#                 global_min_frame = min(global_min_frame, frame_id)
#                 global_max_frame = max(global_max_frame, frame_id)

#                 pts = points[conf_mask]
#                 if pts.shape[0] == 0:
#                     continue
#                 voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
#                 unique_voxels = set(map(tuple, voxel_idx))
#                 for v in unique_voxels:
#                     if v not in voxel_to_frames:
#                         voxel_to_frames[v] = set()
#                     voxel_to_frames[v].add(frame_id)

#         total_frame_span = max(global_max_frame - global_min_frame, 1e-6)

#         static_voxels = set()
#         for v, frames in voxel_to_frames.items():
#             fmin, fmax = min(frames), max(frames)
#             observed_span = max(fmax - fmin, 1e-6)
#             coverage = observed_span / total_frame_span
#             # 密度: 在它出现的这段时间跨度里,到底有多大比例的帧真的看到了它
#             density = len(frames) / observed_span if observed_span > 0 else 1.0

#             # 静态结构:span够长 且 span内观测密度够高(持续存在,不是来回经过两次)
#             if coverage >= min_temporal_coverage and density >= min_density:
#                 static_voxels.add(v)

#         return static_voxels


    
#     def get_largest_key(self, ignore_loop_closure_submaps=False):
#         """
#         Get the largest key of the first node of any submap.
#         Return: The largest key, or None if the dictionary is empty.
#         """
#         if len(self.submaps) == 0:
#             return None
#         if ignore_loop_closure_submaps:
#             non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
#             return max(non_lc_keys)
#         return max(self.submaps.keys())
    
#     def get_submap(self, id):
#         return self.submaps[id]

#     def get_latest_submap(self, ignore_loop_closure_submaps=False):
#         return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

#     def retrieve_best_semantic_frame(self, query_text_vector):
#         overall_best_score = 0.0
#         overall_best_submap_id = 0
#         overall_best_frame_index = 0
#         # search for best image to target image
#         sorted_keys = sorted(self.submaps.keys())
#         for index, submap_key in enumerate(sorted_keys):
#             submap = self.submaps[submap_key]
#             if submap.get_lc_status():
#                 continue
#             submap_embeddings = submap.get_all_semantic_vectors()
#             scores = []
#             for index, embedding in enumerate(submap_embeddings):
#                 score = cosine_similarity(embedding, query_text_vector)
#                 scores.append(score)
            
#             best_score_id = np.argmax(scores)
#             best_score = scores[best_score_id]

#             if best_score > overall_best_score:
#                 overall_best_score = best_score
#                 overall_best_submap_id = submap_key
#                 overall_best_frame_index = best_score_id

#         return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
#     def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
#         overall_best_score = 1000
#         overall_best_submap_id = 0
#         overall_best_frame_index = 0
#         # search for best image to target image
#         sorted_keys = sorted(self.submaps.keys())
#         for index, submap_key in enumerate(sorted_keys):
#             if submap_key == current_submap_id:
#                 continue

#             if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
#                 continue

#             else:
#                 submap = self.submaps[submap_key]
#                 if submap.get_lc_status():
#                     continue
#                 submap_embeddings = submap.get_all_retrieval_vectors()
#                 scores = []
#                 for index, embedding in enumerate(submap_embeddings):
#                     score = torch.linalg.norm(embedding-query_vector)
#                     # score = embedding @ query_vector.t()
#                     scores.append(score.item())

#                 # for now assume we can only have at most one loop closure per submap
                
#                 best_score_id = np.argmin(scores)
#                 best_score = scores[best_score_id]

#                 if best_score < overall_best_score:
#                     overall_best_score = best_score
#                     overall_best_submap_id = submap_key
#                     overall_best_frame_index = best_score_id

#         return overall_best_score, overall_best_submap_id, overall_best_frame_index

#     def get_frames_from_loops(self, loops):
#         frames = []
#         for detected_loop in loops:
#             frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
#         return frames
    
#     def get_submaps(self):
#         return self.submaps.values()

#     def ordered_submaps_by_key(self):
#         for k in sorted(self.submaps):
#             yield self.submaps[k]
    
#     def get_all_homographies(self, graph):
#         homographies = []
#         for submap in self.ordered_submaps_by_key():
#             for pose_num in range(len(submap.poses)):
#                 id = int(submap.get_id() + pose_num)
#                 homographies.append(graph.get_homography(id))
#         return np.stack(homographies)

#     def get_all_cam_matricies(self, graph, give_camera_mat):
#         cam_mats = []
#         for submap in self.ordered_submaps_by_key():
#             if submap.get_lc_status():
#                 continue
#             poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
#             cam_mats.append(poses)
#         return np.vstack(cam_mats)

#     def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
#         all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
#         with open(file_name, "w") as f:

#             if self.rectifying_H_mats:
#                 assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
#                 print("Using rectifying homographies when writing poses to file.")
#             count = 0
#             for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
#                 if submap.get_lc_status():
#                     continue
#                 frame_ids = submap.get_frame_ids()
#                 print(frame_ids)
#                 for frame_index, frame_id in enumerate(frame_ids):
#                     pose = all_poses[count]
#                     K, rotation_matrix, t, scale = decompose_camera(pose)
#                     # print("Decomposed K:\n", K)
#                     count += 1
#                     x, y, z = t
#                     if kitti_format:
#                         pose_matrix = np.eye(4)
#                         pose_matrix[:3, :3] = rotation_matrix
#                         pose_matrix[:3, 3] = t
#                         output = pose_matrix.flatten()[:-4]
#                         output = np.array([float(frame_id), *output])
#                     else:
#                         quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
#                         output = np.array([float(frame_id), x, y, z, *quaternion])
#                     f.write(" ".join(f"{v:.8f}" for v in output) + "\n")    

#     def write_points_to_file(self, graph, file_name, static_voxels=None, voxel_size=None):
#         pcd_all = []
#         colors_all = []
#         for submap in self.ordered_submaps_by_key():
#             points, colors = submap.get_points_and_colors_in_world_frame(graph, static_voxels=static_voxels, voxel_size=voxel_size)
#             if points is None or points.shape[0] == 0:
#                 continue
#             pcd_all.append(points.reshape(-1, 3))
#             colors_all.append(colors.reshape(-1, 3))
#         pcd_all = np.concatenate(pcd_all, axis=0)
#         colors_all = np.concatenate(colors_all, axis=0)
#         if colors_all.max() > 1.0:
#             colors_all = colors_all / 255.0
#         pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
#         pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
#         o3d.io.write_point_cloud(file_name, pcd_all)

#     def export_temporal_debug_pointcloud(self, graph, out_path, voxel_size=0.1):
#         """
#         Debug utility: colors every (confidence-filtered) point in the map by its
#         voxel's temporal uncertainty score. Red = high uncertainty (narrow
#         frame_id window -> suspected dynamic/ghost), Blue = low uncertainty (wide
#         frame_id window -> suspected static). Writes a colored .pcd plus raw
#         per-voxel stats (span, observation count, uncertainty) to an .npz for
#         numeric inspection.
#         """
#         voxel_to_frame_range = {}
#         voxel_to_count = defaultdict(int)
#         global_min_frame = float("inf")
#         global_max_frame = float("-inf")

#         all_points = []
#         all_voxel_keys = []

#         for submap in self.ordered_submaps_by_key():
#             point_list, frame_id_list, frame_conf_mask = submap.get_points_list_in_world_frame(graph)
#             for points, frame_id, conf_mask in zip(point_list, frame_id_list, frame_conf_mask):
#                 global_min_frame = min(global_min_frame, frame_id)
#                 global_max_frame = max(global_max_frame, frame_id)

#                 pts = points[conf_mask]
#                 if pts.shape[0] == 0:
#                     continue
#                 voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
#                 unique_voxels = set(map(tuple, voxel_idx))
#                 for v in unique_voxels:
#                     voxel_to_count[v] += 1
#                     if v not in voxel_to_frame_range:
#                         voxel_to_frame_range[v] = [frame_id, frame_id]
#                     else:
#                         rng = voxel_to_frame_range[v]
#                         if frame_id < rng[0]:
#                             rng[0] = frame_id
#                         if frame_id > rng[1]:
#                             rng[1] = frame_id

#                 all_points.append(pts)
#                 all_voxel_keys.append(voxel_idx)

#         all_points = np.concatenate(all_points, axis=0)
#         all_voxel_keys = np.concatenate(all_voxel_keys, axis=0)
#         total_frame_span = max(global_max_frame - global_min_frame, 1e-6)

#         keys = list(voxel_to_frame_range.keys())
#         key_to_idx = {k: i for i, k in enumerate(keys)}
#         spans = np.zeros(len(keys))
#         counts = np.zeros(len(keys))
#         for k, (fmin, fmax) in voxel_to_frame_range.items():
#             i = key_to_idx[k]
#             spans[i] = fmax - fmin
#             counts[i] = voxel_to_count[k]

#         uncertainty = 1.0 - (spans / total_frame_span)

#         print(f"Total frame span: {total_frame_span:.0f}")
#         print(f"Num voxels: {len(keys)}")
#         print("Span percentiles (frames) [5,25,50,75,95]:", np.percentile(spans, [5, 25, 50, 75, 95]))
#         print("Count percentiles (distinct frames observing voxel) [5,25,50,75,95]:", np.percentile(counts, [5, 25, 50, 75, 95]))
#         print("Uncertainty percentiles [5,25,50,75,95]:", np.percentile(uncertainty, [5, 25, 50, 75, 95]))

#         point_voxel_idx = np.array([key_to_idx[tuple(v)] for v in all_voxel_keys])
#         point_uncertainty = uncertainty[point_voxel_idx]

#         colors = np.zeros((len(all_points), 3))
#         colors[:, 0] = point_uncertainty        # red = high uncertainty (suspected dynamic)
#         colors[:, 2] = 1.0 - point_uncertainty  # blue = low uncertainty (suspected static)

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(all_points)
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#         o3d.io.write_point_cloud(out_path, pcd)
#         print(f"Wrote debug-colored point cloud to {out_path}")

#         stats_path = os.path.splitext(out_path)[0] + "_stats.npz"
#         np.savez(stats_path, voxel_coords=np.array(keys), spans=spans, counts=counts,
#                 uncertainty=uncertainty, total_frame_span=total_frame_span)
#         print(f"Wrote raw per-voxel stats to {stats_path}")

class GraphMap:
    def __init__(self):
        self.submaps = dict()
        self.rectifying_H_mats = []
        self.non_lc_submap_ids = []
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue

            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for index, embedding in enumerate(submap_embeddings):
                    score = torch.linalg.norm(embedding-query_vector)
                    # score = embedding @ query_vector.t()
                    scores.append(score.item())

                # for now assume we can only have at most one loop closure per submap
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        return frames
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]
    
    def get_all_homographies(self, graph):
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    x, y, z = t
                    if kitti_format:
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]
                        output = np.array([float(frame_id), *output])
                    else:
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")    

    # def write_points_to_file(self, graph, file_name):
    #     pcd_all = []
    #     colors_all = []
    #     for submap in self.ordered_submaps_by_key():
    #         pcd = submap.get_points_in_world_frame(graph)
    #         pcd = pcd.reshape(-1, 3)
    #         pcd_all.append(pcd)
    #         colors_all.append(submap.get_points_colors())
    #     pcd_all = np.concatenate(pcd_all, axis=0)
    #     colors_all = np.concatenate(colors_all, axis=0)
    #     if colors_all.max() > 1.0:
    #         colors_all = colors_all / 255.0
    #     pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
    #     pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
    #     o3d.io.write_point_cloud(file_name, pcd_all)
    
    def write_points_to_file(self, graph, file_name, static_voxels=None, voxel_size=None):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            points, colors = submap.get_points_and_colors_in_world_frame(graph, static_voxels=static_voxels, voxel_size=voxel_size)
            if points is None or points.shape[0] == 0:
                continue
            pcd_all.append(points.reshape(-1, 3))
            colors_all.append(colors.reshape(-1, 3))
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)

    
    # def compute_static_voxel_set(self, graph, voxel_size=0.1, min_temporal_coverage=0.1, min_density=0.5):
    #     voxel_to_frames = {}
    #     global_min_frame = float("inf")
    #     global_max_frame = float("-inf")

    #     for submap in self.ordered_submaps_by_key():
    #         point_list, frame_id_list, frame_conf_mask = submap.get_points_list_in_world_frame(graph)
    #         for points, frame_id, conf_mask in zip(point_list, frame_id_list, frame_conf_mask):
    #             global_min_frame = min(global_min_frame, frame_id)
    #             global_max_frame = max(global_max_frame, frame_id)

    #             pts = points[conf_mask]
    #             if pts.shape[0] == 0:
    #                 continue
    #             voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    #             unique_voxels = set(map(tuple, voxel_idx))
    #             for v in unique_voxels:
    #                 if v not in voxel_to_frames:
    #                     voxel_to_frames[v] = set()
    #                 voxel_to_frames[v].add(frame_id)

    #     total_frame_span = max(global_max_frame - global_min_frame, 1e-6)

    #     static_voxels = set()
    #     for v, frames in voxel_to_frames.items():
    #         fmin, fmax = min(frames), max(frames)
    #         observed_span = max(fmax - fmin, 1e-6)
    #         coverage = observed_span / total_frame_span
    #         density = len(frames) / observed_span if observed_span > 0 else 1.0

    #         if coverage >= min_temporal_coverage and density >= min_density:
    #             static_voxels.add(v)

    #     return static_voxels

    def compute_static_voxel_set(self, graph, voxel_size=0.1, min_temporal_coverage=0.1, min_density=0.5):
        """
            NOTE: uses each frame's global keyframe index (submap.get_id() + index
            within the submap, matching the pose graph's node numbering) as the time
            axis, not the raw video frame_id. Keyframes are sparsely and unevenly
            sampled (optical-flow driven), so raw frame_id gaps don't reflect actual
            observation density -- using them here made coverage/density nearly
            impossible to satisfy for genuinely static voxels.
        """
        voxel_to_frames = {}
        global_min_idx = float("inf")
        global_max_idx = float("-inf")

        for submap in self.ordered_submaps_by_key():
            point_list, _, frame_conf_mask = submap.get_points_list_in_world_frame(graph)
            for index, (points, conf_mask) in enumerate(zip(point_list, frame_conf_mask)):
                global_idx = submap.get_id() + index
                global_min_idx = min(global_min_idx, global_idx)
                global_max_idx = max(global_max_idx, global_idx)

                pts = points[conf_mask]
                if pts.shape[0] == 0:
                    continue
                voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
                unique_voxels = set(map(tuple, voxel_idx))
                for v in unique_voxels:
                    if v not in voxel_to_frames:
                        voxel_to_frames[v] = set()
                    voxel_to_frames[v].add(global_idx)

        total_frame_span = max(global_max_idx - global_min_idx, 1e-6)

        static_voxels = set()
        for v, frames in voxel_to_frames.items():
            fmin, fmax = min(frames), max(frames)
            observed_span = max(fmax - fmin, 1e-6)
            coverage = observed_span / total_frame_span
            density = len(frames) / observed_span if observed_span > 0 else 1.0

            if coverage >= min_temporal_coverage and density >= min_density:
                static_voxels.add(v)

        return static_voxels


    def export_temporal_debug_pointcloud(self, graph, out_path, voxel_size=0.1):
        """
        Debug utility: colors every (confidence-filtered) point by its voxel's
        temporal uncertainty score. Red = high uncertainty (suspected dynamic),
        Blue = low uncertainty (suspected static). Also writes per-voxel stats
        to an .npz.
        """
        voxel_to_frame_range = {}
        voxel_to_count = defaultdict(int)
        global_min_frame = float("inf")
        global_max_frame = float("-inf")

        all_points = []
        all_voxel_keys = []

        # for submap in self.ordered_submaps_by_key():
        #     point_list, frame_id_list, frame_conf_mask = submap.get_points_list_in_world_frame(graph)
        #     for points, frame_id, conf_mask in zip(point_list, frame_id_list, frame_conf_mask):
        #         global_min_frame = min(global_min_frame, frame_id)
        #         global_max_frame = max(global_max_frame, frame_id)

        #         pts = points[conf_mask]
        #         if pts.shape[0] == 0:
        #             continue
        #         voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
        #         unique_voxels = set(map(tuple, voxel_idx))
        #         for v in unique_voxels:
        #             voxel_to_count[v] += 1
        #             if v not in voxel_to_frame_range:
        #                 voxel_to_frame_range[v] = [frame_id, frame_id]
        #             else:
        #                 rng = voxel_to_frame_range[v]
        #                 if frame_id < rng[0]:
        #                     rng[0] = frame_id
        #                 if frame_id > rng[1]:
        #                     rng[1] = frame_id

        #         all_points.append(pts)
        #         all_voxel_keys.append(voxel_idx)

        for submap in self.ordered_submaps_by_key():
            point_list, _, frame_conf_mask = submap.get_points_list_in_world_frame(graph)
            for index, (points, conf_mask) in enumerate(zip(point_list, frame_conf_mask)):
                global_idx = submap.get_id() + index
                global_min_frame = min(global_min_frame, global_idx)
                global_max_frame = max(global_max_frame, global_idx)

                pts = points[conf_mask]
                if pts.shape[0] == 0:
                    continue
                voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
                unique_voxels = set(map(tuple, voxel_idx))
                for v in unique_voxels:
                    voxel_to_count[v] += 1
                    if v not in voxel_to_frame_range:
                        voxel_to_frame_range[v] = [global_idx, global_idx]
                    else:
                        rng = voxel_to_frame_range[v]
                        if global_idx < rng[0]:
                            rng[0] = global_idx
                        if global_idx > rng[1]:
                            rng[1] = global_idx

                all_points.append(pts)
                all_voxel_keys.append(voxel_idx)

        
        all_points = np.concatenate(all_points, axis=0)
        all_voxel_keys = np.concatenate(all_voxel_keys, axis=0)
        total_frame_span = max(global_max_frame - global_min_frame, 1e-6)

        keys = list(voxel_to_frame_range.keys())
        key_to_idx = {k: i for i, k in enumerate(keys)}
        spans = np.zeros(len(keys))
        counts = np.zeros(len(keys))
        for k, (fmin, fmax) in voxel_to_frame_range.items():
            i = key_to_idx[k]
            spans[i] = fmax - fmin
            counts[i] = voxel_to_count[k]

        uncertainty = 1.0 - (spans / total_frame_span)

        print(f"Total frame span: {total_frame_span:.0f}")
        print(f"Num voxels: {len(keys)}")
        print("Span percentiles [5,25,50,75,95]:", np.percentile(spans, [5, 25, 50, 75, 95]))
        print("Count percentiles [5,25,50,75,95]:", np.percentile(counts, [5, 25, 50, 75, 95]))
        print("Uncertainty percentiles [5,25,50,75,95]:", np.percentile(uncertainty, [5, 25, 50, 75, 95]))

        point_voxel_idx = np.array([key_to_idx[tuple(v)] for v in all_voxel_keys])
        point_uncertainty = uncertainty[point_voxel_idx]

        colors = np.zeros((len(all_points), 3))
        colors[:, 0] = point_uncertainty
        colors[:, 2] = 1.0 - point_uncertainty

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"Wrote debug-colored point cloud to {out_path}")

        stats_path = os.path.splitext(out_path)[0] + "_stats.npz"
        np.savez(stats_path, voxel_coords=np.array(keys), spans=spans, counts=counts,
                uncertainty=uncertainty, total_frame_span=total_frame_span)
        print(f"Wrote raw per-voxel stats to {stats_path}")
