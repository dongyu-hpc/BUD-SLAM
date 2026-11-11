# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_dynamic_filter --dynamic_conf_threshold 10
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_temporal_filter --temporal_voxel_size 0.1 --dynamic_conf_threshold 10 --min_submap_support 2
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_temporal_filter --temporal_voxel_size 0.1 --min_temporal_coverage 0.1
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_temporal_filter --temporal_voxel_size 0.1 --min_temporal_coverage 0.1 --dump_temporal_debug
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_temporal_filter --temporal_voxel_size 0.1 --min_temporal_coverage 0.1 --min_density 0.5 --dump_temporal_debug
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_temporal_filter --temporal_voxel_size 0.1 --min_temporal_coverage 0.1 --min_density 0.5 --dump_temporal_debug --enable_dynamic_split
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt --enable_dynamic_split
# python3 main.py --image_folder tesse_cd_apartment --max_loops 1 --log_results --log_path outputs/apartment.txt
# python3 main.py --image_folder data/rgbd_dataset_freiburg2_desk_with_person/rgb --max_loops 1 --log_results --log_path outputs/freiburg.txt

import os
import glob
import time
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap

from vggt4d.models.vggt4d import VGGTFor4D
# from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--enable_dynamic_filter", action="store_true", help="Enable stricter confidence-based filtering to remove likely-dynamic points from the map")
parser.add_argument("--dynamic_conf_threshold", type=float, default=70.0, help="Confidence percentile threshold used when --enable_dynamic_filter is set (higher = more aggressive filtering)")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--enable_temporal_filter", action="store_true", help="Remove points with high temporal uncertainty (observed only in a narrow time window), i.e. likely dynamic objects")
parser.add_argument("--temporal_voxel_size", type=float, default=0.1, help="Voxel size (meters) for temporal uncertainty estimation")
parser.add_argument("--dump_temporal_debug", action="store_true", help="Export a debug point cloud colored by temporal uncertainty (red=suspected dynamic, blue=suspected static) plus per-voxel stats, for diagnosing filter behavior")
parser.add_argument("--min_temporal_coverage", type=float, default=0.1, help="Minimum fraction (0-1) of the trajectory's total frame span a voxel must be observed across to be kept as static")
parser.add_argument("--min_density", type=float, default=0.5, help="Minimum fraction (0-1) of frames within a voxel's observed time span that must actually see it, to be kept as static. Filters out voxels revisited sparsely (e.g. a person passing through twice) that would otherwise pass the coverage check")
parser.add_argument("--enable_dynamic_split", action="store_true",
    help="Split points into static (persistent map) vs dynamic (sliding window) "
         "using disagreement between depth_head and point_head 3D estimates. "
         "Off by default: relies on an unverified assumption that both heads' "
         "outputs share the same per-submap coordinate frame.")
parser.add_argument("--dynamic_residual_thresh", type=float, default=0.08,
    help="Relative disagreement threshold (fraction of point distance from "
         "the submap origin) between depth_head and point_head estimates, "
         "above which a point is flagged dynamic.")
parser.add_argument("--dynamic_window_frames", type=int, default=5,
    help="Number of most recent frames' dynamic points kept visible at once "
         "in the sliding-window dynamic point cloud.")

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        dynamic_conf_threshold=args.dynamic_conf_threshold if args.enable_dynamic_filter else None,
        enable_dynamic_split=args.enable_dynamic_split,
        dynamic_residual_thresh=args.dynamic_residual_thresh,
        dynamic_window_frames=args.dynamic_window_frames,
    )

    print("Initializing and loading VGGT model...")


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = VGGTFor4D()
    _URL = "https://hf-mirror.com/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                if img is None:
                    print(f"Failed to read image: {image_name}")
                    continue
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)


        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            solver.update_dynamic_window()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    static_voxels = None
    temporal_voxel_size = None
    if args.enable_temporal_filter:
        print("Computing static voxel set from temporal coverage/density...")
        temporal_voxel_size = args.temporal_voxel_size
        static_voxels = solver.map.compute_static_voxel_set(
            solver.graph,
            voxel_size=temporal_voxel_size,
            min_temporal_coverage=args.min_temporal_coverage,
            min_density=args.min_density,
        )
        print(f"Static voxel set size: {len(static_voxels)}")

        if args.dump_temporal_debug:
            debug_path = args.log_path.replace(".txt", "_temporal_debug.pcd")
            solver.map.export_temporal_debug_pointcloud(solver.graph, debug_path, voxel_size=temporal_voxel_size)


    if args.run_os:
        # Register the viser object-query panel so the user can search for
        # objects in the viewer in addition to the terminal prompt below.
        import threading
        data_lock = threading.Lock()
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return

            if query == "q":
                print("Exiting.")
                return

            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())


            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    # if not args.vis_map:
    #     # just show the map after all submaps have been processed
    #     solver.update_all_submap_vis()

    if not args.vis_map:
        solver.update_all_submap_vis(static_voxels=static_voxels, voxel_size=temporal_voxel_size)


    # if args.log_results:
    #     solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

    #     if not args.skip_dense_log:
    #         # Log the full point cloud as one file
    #         solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        if not args.skip_dense_log:
            solver.map.write_points_to_file(
                solver.graph,
                args.log_path.replace(".txt", "_points.pcd"),
                static_voxels=static_voxels,
                voxel_size=temporal_voxel_size,
            )



if __name__ == "__main__":
    main()
