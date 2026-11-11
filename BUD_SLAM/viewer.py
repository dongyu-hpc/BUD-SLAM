import time
from typing import Dict, List
import numpy as np
import torch
import viser
import viser.transforms as viser_tf
# 调用链条：main <- solver <- viewer
import threading

class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # --- GUI Elements ---
        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        self.gui_show_points = self.server.gui.add_checkbox("Show Points", initial_value=True)
        self.gui_show_points.on_update(self._on_update_show_points)

        # Add a button to trigger the walkthrough
        self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
        self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        # Point cloud state: each submap gets one anchor frame, and one or
        # more named point cloud handles parented under that anchor. Moving
        # the anchor (e.g. on loop closure) moves all clouds under it for
        # free on the frontend, without re-transforming point arrays here.
        self.submap_anchors: Dict[int, viser.FrameHandle] = {}
        self.submap_points: Dict[int, Dict[str, viser.PointCloudHandle]] = {}

        num_rand_colors = 250
        np.random.seed(100)
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
        self.submap_id_to_color = dict()
        self.obj_id = 0

        self.dynamic_frame_handles: Dict[int, viser.PointCloudHandle] = {}
        self.dynamic_frame_indices: List[int] = []
        self.gui_dyn_playing = None
        self.gui_dyn_timestep = None
        self.gui_dyn_window = None
        self.gui_dyn_fps = None


    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W) or None to show frustums without the camera images
        """

        if images_ is not None and isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            next_id = len(self.submap_id_to_color) + 1
            self.submap_id_to_color[submap_id] = next_id
        self.submap_frames[submap_id] = []
        self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum. When images are not provided, show
            # just the frustum (no image) with a default FOV/aspect to keep the
            # visualization fast.
            fov = np.radians(90)  # default FOV if no images provided
            img = None
            aspect = 1.0
            if images_ is not None:
                img = images_[img_id]
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                h, w = img.shape[:2]
                fy = 1.1 * h
                fov = 2 * np.arctan2(h / 2, fy)
                aspect = w / h

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=aspect,
                scale=0.05,
                image=img,
                line_width=3.0,
                color=self.random_colors[self.submap_id_to_color[submap_id]]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def visualize_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        submap_id: int,
        key: str = "points",
        point_size: float = 0.01,
        point_shape: str = "rounded",
    ) -> None:
        """
        Add or update a point cloud belonging to a submap.

        points: (N, 3) world-space coordinates at insertion time
        colors: (N, 3)
        submap_id: which submap this cloud belongs to
        key: distinguishes multiple clouds under the same submap, e.g.
             "background" / "foreground", or a per-frame key like
             f"frame_{img_id}" for dynamic content that changes within
             the same submap.

        The cloud is parented under a per-submap anchor frame. This means
        `update_submap_pose` can move the whole cloud for free (on the
        frontend) when loop closure corrects the submap's global pose,
        without re-transforming the point array in Python.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()

        anchor_name = f"submap_{submap_id}/anchor"
        if submap_id not in self.submap_anchors:
            anchor = self.server.scene.add_frame(
                anchor_name,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                show_axes=False,
            )
            self.submap_anchors[submap_id] = anchor
            self.submap_points[submap_id] = {}

        existing = self.submap_points[submap_id].get(key)
        if existing is not None:
            # Cheaper than re-adding: just push new data into the same handle.
            existing.points = points
            existing.colors = colors
            return

        pts_handle = self.server.scene.add_point_cloud(
            f"{anchor_name}/{key}",
            points=points,
            colors=colors,
            point_size=point_size,
            point_shape=point_shape,
        )
        pts_handle.visible = self.gui_show_points.value
        self.submap_points[submap_id][key] = pts_handle

    def update_submap_pose(self, submap_id: int, wxyz: np.ndarray, position: np.ndarray) -> None:
        """
        Apply a corrected world pose to an entire submap (e.g. after loop closure).

        Only moves this submap's point cloud anchor. Camera frames/frustums
        for the same submap are stored as absolute world poses and are not
        parented under the anchor, so they still need to be updated wherever
        your existing loop-closure code already corrects
        `self.submap_frames[submap_id]` handles.
        """
        anchor = self.submap_anchors.get(submap_id)
        if anchor is None:
            return
        anchor.wxyz = wxyz
        anchor.position = position

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def _on_update_show_points(self, _) -> None:
        """Toggle visibility of all point clouds across all submaps."""
        visible = self.gui_show_points.value
        for cloud_dict in self.submap_points.values():
            for h in cloud_dict.values():
                h.visible = visible

    def visualize_obb(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        rotation: np.ndarray,
        color = (255, 0, 0),
        line_width: float = 2.0,
    ):
        """
        Visualize an oriented bounding box (OBB) in Viser.

        Parameters
        ----------
        name : str
            Identifier for the OBB in the scene.
        center : (3,) array
            World-space center of the OBB.
        extent : (3,) array
            Full side lengths of the OBB (dx, dy, dz).
        rotation : (3,3) array
            Rotation matrix of the OBB in world coordinates.
        color : tuple[int,int,int]
            RGB color of the wireframe box.
        line_width : float
            Thickness of the box edges.

        Notes
        -----
        The box is drawn as a wireframe with 12 edges.
        """

        # Compute local corners (8)
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ], dtype=np.float32)  # shape (8,3)

        # Transform to world
        corners_world = (rotation @ corners_local.T).T + center  # shape (8,3)

        # Build edges (12 line segments) as start/end pairs
        edges_idx = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7)   # vertical edges
        ]

        segments = []
        for (i,j) in edges_idx:
            segments.append(corners_world[i])
            segments.append(corners_world[j])
        # segments is list of length 24, reshape into (N,2,3)
        segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

        name = f"obb_{self.obj_id}"
        self.obj_id += 1
        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=color,       # single color for all segments
            line_width=line_width,
            visible=True
        )

    def add_object_query_gui(self, solver, clip_model, clip_tokenizer, processor, data_lock) -> None:
        """Add an object query panel to the viser sidebar.

        Mirrors the terminal open-set query flow: retrieve the best-matching
        keyframe via CLIP, run SAM3 to segment the queried object, then draw an
        oriented bounding box per detected instance in the 3-D scene. Requires
        --run_os (clip_model / processor loaded) and a non-empty map.
        """
        import torch
        from torchvision.transforms.functional import to_pil_image
        import vggt_slam.slam_utils as utils

        with self.server.gui.add_folder("Object Query"):
            gui_query = self.server.gui.add_text("Query", initial_value="")
            gui_status = self.server.gui.add_markdown("*Enter a query and press Search.*")
            btn_search = self.server.gui.add_button("Search", color="green")

        @btn_search.on_click
        def _on_search(_) -> None:
            query = gui_query.value.strip()
            if not query:
                gui_status.content = "*Enter a query first.*"
                return
            if solver.map.get_num_submaps() == 0:
                gui_status.content = "*No map data yet. Capture some frames first.*"
                return
            gui_status.content = f"*Searching for '{query}'…*"
            try:
                with data_lock:
                    text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
                    best_score, best_submap_id, best_frame_index = \
                        solver.map.retrieve_best_semantic_frame(text_emb)
                    found_submap = solver.map.get_submap(best_submap_id)
                    best_img = found_submap.get_frame_at_index(best_frame_index)

                    with torch.no_grad():
                        pil_img = to_pil_image(best_img)
                        inference_state = processor.set_image(pil_img)
                        output = processor.set_text_prompt(state=inference_state, prompt=query)
                        masks = output["masks"]

                    n = masks.shape[0]
                    for i in range(n):
                        mask = masks[i].cpu().numpy()
                        obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
                            found_submap.get_points_in_mask(best_frame_index, mask, solver.graph)
                        )
                        self.visualize_obb(
                            center=obb_center,
                            extent=obb_extent,
                            rotation=obb_rotation,
                            color=(255, 0, 0),
                            line_width=8.0,
                        )

                if n == 0:
                    gui_status.content = f"*No instances found for '{query}'.*"
                else:
                    gui_status.content = f"**Found {n} instance(s)** for *'{query}'*"
            except Exception as e:
                import traceback
                traceback.print_exc()
                gui_status.content = f"*Error: {e}*"

    def run_walkthrough(self, fps: float = 20.0):
        """
        Walks through the map using the current live positions of all frames.
        This accounts for loop closures because it pulls data from the scene handles.
        """
        # 1. Gather all submap IDs and sort them to ensure a logical sequence
        sorted_submap_ids = sorted(self.submap_frames.keys())

        if not sorted_submap_ids:
            print("No frames found to walk through.")
            return

        clients = self.server.get_clients()
        if not clients:
            print("No clients connected to perform walkthrough.")
            return

        print("Starting walkthrough of updated poses...")

        for sub_id in sorted_submap_ids:
            frames = self.submap_frames[sub_id]
            # Assumes frames were added in chronological order to the list
            for frame_handle in frames:
                # Get the current world-space pose from the visualizer
                # If a loop closure moved the submap, these values will be updated
                current_pos = frame_handle.position
                current_wxyz = frame_handle.wxyz

                # Update all connected clients
                for client in clients.values():
                    client.camera.position = current_pos
                    client.camera.wxyz = current_wxyz

                # Control speed (1/fps)
                time.sleep(1.0 / fps)

    def setup_dynamic_playback_gui(self, window_frames: int = 5, fps: float = 10.0) -> None:
        """
        Playback controls for scrubbing through the history of per-frame
        dynamic (moving-object) point clouds. Call once, only when dynamic
        point splitting is enabled.

        Unlike the static map, dynamic points are kept as one point cloud
        handle per frame (see add_dynamic_frame_points), not merged into a
        single overwritten cloud, so the full history can be scrubbed after
        the fact instead of only showing the latest sliding window.
        """
        with self.server.gui.add_folder("Dynamic Playback"):
            self.gui_dyn_playing = self.server.gui.add_checkbox("Playing", False)
            self.gui_dyn_timestep = self.server.gui.add_slider(
                "Timestep", min=0, max=0, step=1, initial_value=0)
            self.gui_dyn_window = self.server.gui.add_number(
                "Window size", min=1, max=200, step=1, initial_value=window_frames)
            self.gui_dyn_fps = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=1, initial_value=fps)

        @self.gui_dyn_timestep.on_update
        def _(_) -> None:
            self._apply_dynamic_window_visibility(self.gui_dyn_timestep.value, self.gui_dyn_window.value)

        @self.gui_dyn_window.on_update
        def _(_) -> None:
            self._apply_dynamic_window_visibility(self.gui_dyn_timestep.value, self.gui_dyn_window.value)

        def _playback_loop():
            while True:
                if self.gui_dyn_playing.value and self.dynamic_frame_indices:
                    max_idx = self.dynamic_frame_indices[-1]
                    next_val = self.gui_dyn_timestep.value + 1
                    self.gui_dyn_timestep.value = next_val if next_val <= max_idx else 0
                time.sleep(1.0 / max(self.gui_dyn_fps.value, 1))

        threading.Thread(target=_playback_loop, daemon=True).start()

    def add_dynamic_frame_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        global_frame_idx: int,
        point_size: float = 0.005,
    ) -> None:
        """
        Add (or update) the dynamic point cloud for one global frame index.
        Kept as a separate handle per frame so the full history is available
        for playback scrubbing.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()

        name = f"dynamic/frame_{global_frame_idx:06d}"
        existing = self.dynamic_frame_handles.get(global_frame_idx)
        if existing is not None:
            existing.points = points
            existing.colors = colors
            return

        h = self.server.scene.add_point_cloud(
            name, points=points, colors=colors,
            point_size=point_size, point_shape="rounded",
        )
        h.visible = False
        self.dynamic_frame_handles[global_frame_idx] = h
        self.dynamic_frame_indices.append(global_frame_idx)

        if self.gui_dyn_timestep is not None:
            self.gui_dyn_timestep.max = global_frame_idx

    def _apply_dynamic_window_visibility(self, center_idx: int, window_size: int) -> None:
        """Show only frames in (center_idx - window_size, center_idx]."""
        # for idx, h in self.dynamic_frame_handles.items():
        for idx, h in list(self.dynamic_frame_handles.items()):
            h.visible = (center_idx - window_size < idx <= center_idx)

    def set_live_dynamic_window(self, latest_idx: int, window_size: int) -> None:
        """
        Called during active processing to keep the most recent `window_size`
        frames visible and keep the playback slider pinned to the latest
        frame, unless the user is actively scrubbing (Playing checked).
        """
        self._apply_dynamic_window_visibility(latest_idx, window_size)
        if self.gui_dyn_timestep is not None and not (self.gui_dyn_playing and self.gui_dyn_playing.value):
            self.gui_dyn_timestep.value = latest_idx


# import sys
# import time
# from pathlib import Path

# import viser
# from matplotlib import colormaps
# from tqdm import tqdm

# from vggt4d.visualization.scene import Scene4D

# viser_port = 8080

# bg_downsample_factor = 0.1
# point_size = 0.001
# # point_size = 0.02
# fg_downsample_factor = 1.0
# # fg_downsample_factor = 0.5
# camera_frustum_scale = 0.01
# # camera_frustum_scale = 0.1
# axes_scale = 0.05
# cam_line_width = 5
# fps = 24


# if __name__ == "__main__":
#     scene_dir = sys.argv[1]
#     scene_dir = Path(scene_dir)
#     scene = Scene4D(scene_dir)
#     server = viser.ViserServer(port=viser_port)

#     server.scene.set_up_direction('+y')
#     n_frame = scene.num_frame

#     print("Loading frames...")
#     fg_pts_handles = []
#     bg_pts_handles = []
#     frame_handles = []
#     cam_handles = []
#     for i in tqdm(range(n_frame), unit="frame"):
#         frame_handles.append(server.scene.add_frame(
#             f"/scene/frame_{i:04d}", show_axes=False, visible=True))

#         # Load background points for this frame
#         bg_pts, bg_rgb = scene.get_background_points(
#             frame_id=i, downsample_factor=bg_downsample_factor, filter=False)
#         bg_pts_handle = server.scene.add_point_cloud(
#             name=f"/scene/frame_{i:04d}/background",
#             points=bg_pts,
#             colors=bg_rgb,
#             point_size=point_size,
#             point_shape="rounded",
#         )
#         bg_pts_handle.visible = True
#         bg_pts_handles.append(bg_pts_handle)

#         # Load dynamic points for this frame
#         dyn_pts, dyn_rgb = scene.get_dynamic_points(
#             i, downsample_factor=fg_downsample_factor)
#         fg_pts_handle = server.scene.add_point_cloud(
#             name=f"/scene/frame_{i:04d}/foreground",
#             points=dyn_pts,
#             colors=dyn_rgb,
#             point_size=point_size,
#             point_shape="rounded",
#         )
#         fg_pts_handle.visible = (i == 0)  # Only show first frame initially
#         fg_pts_handles.append(fg_pts_handle)

#         fov, aspect, wxyz, pos = scene.get_cam_frustum(i)
#         cmap = colormaps.get_cmap("viridis")
#         color = cmap(i / (n_frame - 1))[:3]
#         cam_handles.append(server.scene.add_camera_frustum(
#             f"/scene/frame_{i:04d}/frustum",
#             fov=fov,
#             scale=camera_frustum_scale,
#             aspect=aspect,
#             wxyz=wxyz,
#             position=pos,
#             color=color,
#             line_width=cam_line_width
#         ))
#         server.scene.add_frame(
#             f"/scene/frame_{i:04d}/frustum/axes",
#             axes_length=camera_frustum_scale * axes_scale * 10,
#             axes_radius=camera_frustum_scale * axes_scale,
#         )

#     with server.gui.add_folder("Playback"):
#         # gui_playing = server.gui.add_checkbox("Playing", False)
#         gui_playing = server.gui.add_checkbox("Playing", True)
#         gui_timestep = server.gui.add_slider(
#             "Timestep",
#             min=0,
#             max=n_frame - 1,
#             step=1,
#             initial_value=0,
#             disabled=True,
#         )
#         gui_framerate = server.gui.add_slider(
#             "FPS", min=1, max=60, step=1, initial_value=fps
#         )
#         gui_point_size = server.gui.add_number(
#             "Point size",
#             min=0.0001,
#             max=0.02,
#             step=0.0002,
#             initial_value=point_size,
#         )

#     with server.gui.add_folder("Background"):
#         gui_show_all_bg = server.gui.add_button("Show All Background")
#         gui_hide_all_bg = server.gui.add_button("Hide All Background")

#     def update_point_size(new_size: float) -> None:
#         new_size = max(0.0004, min(0.02, new_size))
#         with server.atomic():
#             for h in bg_pts_handles:
#                 h.point_size = new_size
#             for h in fg_pts_handles:
#                 h.point_size = new_size

#     @gui_point_size.on_update
#     def _(_) -> None:
#         update_point_size(gui_point_size.value)

#     @gui_timestep.on_update
#     def _(_) -> None:
#         curr_timestep = gui_timestep.value
#         with server.atomic():
#             for i, fg_h in enumerate(fg_pts_handles):
#                 fg_h.visible = i == curr_timestep

#     @gui_show_all_bg.on_click
#     def _(_) -> None:
#         with server.atomic():
#             for bg_h in bg_pts_handles:
#                 bg_h.visible = True

#     @gui_hide_all_bg.on_click
#     def _(_) -> None:
#         with server.atomic():
#             for bg_h in bg_pts_handles:
#                 bg_h.visible = False

#     @gui_playing.on_update
#     def _(_) -> None:
#         gui_timestep.disabled = gui_playing.value

#     # record scene
#     record_fps = 30
#     serializer = server.get_scene_serializer()
#     for i in range(n_frame):
#         with server.atomic():
#             for j, fg_h in enumerate(fg_pts_handles):
#                 fg_h.visible = j == i
#             for j, cam_h in enumerate(cam_handles):
#                 cam_h.visible = j == i
#             serializer.insert_sleep(1.0 / record_fps)
#     record_data = serializer.serialize()
#     with open(scene_dir / "record.viser", "wb") as f:
#         f.write(record_data)

#     while True:
#         if gui_playing.value:
#             gui_timestep.value = (gui_timestep.value + 1) % n_frame
#         time.sleep(1.0 / gui_framerate.value)




# import time
# from typing import Dict, List
# import numpy as np
# import torch
# import viser
# import viser.transforms as viser_tf
# # 调用链条：main <- solver <- viewer


# class Viewer:
#     def __init__(self, port: int = 8080):
#         print(f"Starting viser server on port {port}")

#         self.server = viser.ViserServer(host="0.0.0.0", port=port)
#         self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

#         # --- GUI Elements ---
#         self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
#         self.gui_show_frames.on_update(self._on_update_show_frames)

#         # Add a button to trigger the walkthrough
#         self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
#         self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

#         self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
#         self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

#         num_rand_colors = 250
#         np.random.seed(100)
#         self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
#         self.submap_id_to_color = dict()
#         self.obj_id = 0

#     def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
#         """
#         Add camera frames and frustums to the scene for a specific submap.
#         extrinsics: (S, 3, 4)
#         images_:    (S, 3, H, W) or None to show frustums without the camera images
#         """

#         if images_ is not None and isinstance(images_, torch.Tensor):
#             images_ = images_.cpu().numpy()

#         if submap_id not in self.submap_frames:
#             next_id = len(self.submap_id_to_color) + 1
#             self.submap_id_to_color[submap_id] = next_id
#         self.submap_frames[submap_id] = []
#         self.submap_frustums[submap_id] = []

#         S = extrinsics.shape[0]
#         for img_id in range(S):
#             cam2world_3x4 = extrinsics[img_id]
#             T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

#             frame_name = f"submap_{submap_id}/frame_{img_id}"
#             frustum_name = f"{frame_name}/frustum"

#             # Add the coordinate frame
#             frame_axis = self.server.scene.add_frame(
#                 frame_name,
#                 wxyz=T_world_camera.rotation().wxyz,
#                 position=T_world_camera.translation(),
#                 axes_length=0.05,
#                 axes_radius=0.002,
#                 origin_radius=0.002,
#             )
#             frame_axis.visible = self.gui_show_frames.value
#             self.submap_frames[submap_id].append(frame_axis)

#             # Convert image and add frustum. When images are not provided, show
#             # just the frustum (no image) with a default FOV/aspect to keep the
#             # visualization fast.
#             fov = np.radians(90)  # default FOV if no images provided
#             img = None
#             aspect = 1.0
#             if images_ is not None:
#                 img = images_[img_id]
#                 img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
#                 h, w = img.shape[:2]
#                 fy = 1.1 * h
#                 fov = 2 * np.arctan2(h / 2, fy)
#                 aspect = w / h

#             frustum = self.server.scene.add_camera_frustum(
#                 frustum_name,
#                 fov=fov,
#                 aspect=aspect,
#                 scale=0.05,
#                 image=img,
#                 line_width=3.0,
#                 color=self.random_colors[self.submap_id_to_color[submap_id]]
#             )
#             frustum.visible = self.gui_show_frames.value
#             self.submap_frustums[submap_id].append(frustum)

#     def _on_update_show_frames(self, _) -> None:
#         """Toggle visibility of all camera frames and frustums across all submaps."""
#         visible = self.gui_show_frames.value
#         for frames in self.submap_frames.values():
#             for f in frames:
#                 f.visible = visible
#         for frustums in self.submap_frustums.values():
#             for fr in frustums:
#                 fr.visible = visible

#     def visualize_obb(
#         self,
#         center: np.ndarray,
#         extent: np.ndarray,
#         rotation: np.ndarray,
#         color = (255, 0, 0),
#         line_width: float = 2.0,
#     ):
#         """
#         Visualize an oriented bounding box (OBB) in Viser.

#         Parameters
#         ----------
#         name : str
#             Identifier for the OBB in the scene.
#         center : (3,) array
#             World-space center of the OBB.
#         extent : (3,) array
#             Full side lengths of the OBB (dx, dy, dz).
#         rotation : (3,3) array
#             Rotation matrix of the OBB in world coordinates.
#         color : tuple[int,int,int]
#             RGB color of the wireframe box.
#         line_width : float
#             Thickness of the box edges.

#         Notes
#         -----
#         The box is drawn as a wireframe with 12 edges.
#         """

#         # Compute local corners (8)
#         dx, dy, dz = extent / 2.0
#         corners_local = np.array([
#             [-dx, -dy, -dz],
#             [ dx, -dy, -dz],
#             [ dx,  dy, -dz],
#             [-dx,  dy, -dz],
#             [-dx, -dy,  dz],
#             [ dx, -dy,  dz],
#             [ dx,  dy,  dz],
#             [-dx,  dy,  dz],
#         ], dtype=np.float32)  # shape (8,3)

#         # Transform to world
#         corners_world = (rotation @ corners_local.T).T + center  # shape (8,3)

#         # Build edges (12 line segments) as start/end pairs
#         edges_idx = [
#             (0,1),(1,2),(2,3),(3,0),  # bottom face
#             (4,5),(5,6),(6,7),(7,4),  # top face
#             (0,4),(1,5),(2,6),(3,7)   # vertical edges
#         ]

#         segments = []
#         for (i,j) in edges_idx:
#             segments.append(corners_world[i])
#             segments.append(corners_world[j])
#         # segments is list of length 24, reshape into (N,2,3)
#         segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

#         name = f"obb_{self.obj_id}"
#         self.obj_id += 1
#         self.server.scene.add_line_segments(
#             name=name,
#             points=segments,
#             colors=color,       # single color for all segments
#             line_width=line_width,
#             visible=True
#         )

#     def add_object_query_gui(self, solver, clip_model, clip_tokenizer, processor, data_lock) -> None:
#         """Add an object query panel to the viser sidebar.

#         Mirrors the terminal open-set query flow: retrieve the best-matching
#         keyframe via CLIP, run SAM3 to segment the queried object, then draw an
#         oriented bounding box per detected instance in the 3-D scene. Requires
#         --run_os (clip_model / processor loaded) and a non-empty map.
#         """
#         import torch
#         from torchvision.transforms.functional import to_pil_image
#         import vggt_slam.slam_utils as utils

#         with self.server.gui.add_folder("Object Query"):
#             gui_query = self.server.gui.add_text("Query", initial_value="")
#             gui_status = self.server.gui.add_markdown("*Enter a query and press Search.*")
#             btn_search = self.server.gui.add_button("Search", color="green")

#         @btn_search.on_click
#         def _on_search(_) -> None:
#             query = gui_query.value.strip()
#             if not query:
#                 gui_status.content = "*Enter a query first.*"
#                 return
#             if solver.map.get_num_submaps() == 0:
#                 gui_status.content = "*No map data yet. Capture some frames first.*"
#                 return
#             gui_status.content = f"*Searching for '{query}'…*"
#             try:
#                 with data_lock:
#                     text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
#                     best_score, best_submap_id, best_frame_index = \
#                         solver.map.retrieve_best_semantic_frame(text_emb)
#                     found_submap = solver.map.get_submap(best_submap_id)
#                     best_img = found_submap.get_frame_at_index(best_frame_index)

#                     with torch.no_grad():
#                         pil_img = to_pil_image(best_img)
#                         inference_state = processor.set_image(pil_img)
#                         output = processor.set_text_prompt(state=inference_state, prompt=query)
#                         masks = output["masks"]

#                     n = masks.shape[0]
#                     for i in range(n):
#                         mask = masks[i].cpu().numpy()
#                         obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
#                             found_submap.get_points_in_mask(best_frame_index, mask, solver.graph)
#                         )
#                         self.visualize_obb(
#                             center=obb_center,
#                             extent=obb_extent,
#                             rotation=obb_rotation,
#                             color=(255, 0, 0),
#                             line_width=8.0,
#                         )

#                 if n == 0:
#                     gui_status.content = f"*No instances found for '{query}'.*"
#                 else:
#                     gui_status.content = f"**Found {n} instance(s)** for *'{query}'*"
#             except Exception as e:
#                 import traceback
#                 traceback.print_exc()
#                 gui_status.content = f"*Error: {e}*"

#     def run_walkthrough(self, fps: float = 20.0):
#             """
#             Walks through the map using the current live positions of all frames.
#             This accounts for loop closures because it pulls data from the scene handles.
#             """
#             # 1. Gather all submap IDs and sort them to ensure a logical sequence
#             sorted_submap_ids = sorted(self.submap_frames.keys())
            
#             if not sorted_submap_ids:
#                 print("No frames found to walk through.")
#                 return

#             clients = self.server.get_clients()
#             if not clients:
#                 print("No clients connected to perform walkthrough.")
#                 return

#             print("Starting walkthrough of updated poses...")

#             for sub_id in sorted_submap_ids:
#                 frames = self.submap_frames[sub_id]
#                 # Assumes frames were added in chronological order to the list
#                 for frame_handle in frames:
#                     # Get the current world-space pose from the visualizer
#                     # If a loop closure moved the submap, these values will be updated
#                     current_pos = frame_handle.position
#                     current_wxyz = frame_handle.wxyz

#                     # Update all connected clients
#                     for client in clients.values():
#                         client.camera.position = current_pos
#                         client.camera.wxyz = current_wxyz
                    
#                     # Control speed (1/fps)
#                     time.sleep(1.0 / fps)