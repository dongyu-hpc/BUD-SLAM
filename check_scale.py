"""
诊断VGGT单个submap内部输出的度量尺度是否正确。
不经过solver的多submap拼接，直接跑一遍最原始的forward，
检查两点间距离是否符合真实世界尺度。

用法:
    python check_scale.py --image_folder tesse_cd_apartment --num_frames 16 \
        --model_path /home/gdy/.cache/torch/hub/checkpoints/model_tracker_fixed_e20.pt \
        --frame_idx 0 --p1 100 50 --p2 100 400
    
    python check_scale.py \
        --image_folder tesse_cd_apartment \
        --num_frames 16 \
        --model_path /home/gdy/.cache/torch/hub/checkpoints/model_tracker_fixed_e20.pt \
        --frame_idx 0 \
        --p1 100 50 \
        --p2 100 400
        
"""

import argparse
import glob
import os

import numpy as np
import torch


def load_model(model_path, device):
    from vggt4d.models.vggt4d import VGGTFor4D  # 按你实际import路径调整

    model = VGGTFor4D()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = model.to(dtype).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="图片所在目录")
    parser.add_argument("--pattern", type=str, default="*.jpg")
    parser.add_argument("--num_frames", type=int, default=16, help="取前N帧作为一个submap")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame_idx", type=int, default=0, help="在哪一帧上量距离")
    parser.add_argument("--p1", type=int, nargs=2, required=True, metavar=("ROW", "COL"),
                         help="第一个点的像素坐标 (行, 列) 即 (y, x)")
    parser.add_argument("--p2", type=int, nargs=2, required=True, metavar=("ROW", "COL"),
                         help="第二个点的像素坐标 (行, 列) 即 (y, x)")
    parser.add_argument("--real_distance", type=float, default=None,
                         help="可选: 你目测/已知的这两点真实距离(米), 提供后会自动算出尺度比例")
    args = parser.parse_args()

    from vggt.utils.load_fn import load_and_preprocess_images  # 按你实际路径调整

    image_names = sorted(glob.glob(os.path.join(args.image_folder, args.pattern)))
    if len(image_names) == 0:
        raise ValueError(f"No images found in {args.image_folder}")
    image_names = image_names[:args.num_frames]
    print(f"Using {len(image_names)} frames from {args.image_folder}")

    images = load_and_preprocess_images(image_names).to(args.device)
    print(f"Images tensor shape: {images.shape}")

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, args.device)

    with torch.no_grad():
        predictions, _, _, _ = model(images.to(next(model.parameters()).dtype))

    world_points = predictions["world_points"][0].float().cpu().numpy()  # [S, H, W, 3]
    print(f"world_points shape: {world_points.shape}")

    S, H, W, _ = world_points.shape
    if args.frame_idx >= S:
        raise ValueError(f"frame_idx {args.frame_idx} out of range, only {S} frames")

    r1, c1 = args.p1
    r2, c2 = args.p2
    for name, r, c in [("p1", r1, c1), ("p2", r2, c2)]:
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(f"{name} coordinate ({r}, {c}) out of image bounds ({H}, {W})")

    p1_3d = world_points[args.frame_idx, r1, c1]
    p2_3d = world_points[args.frame_idx, r2, c2]

    print(f"\nframe {args.frame_idx}")
    print(f"p1 pixel=({r1},{c1}) -> world={p1_3d}")
    print(f"p2 pixel=({r2},{c2}) -> world={p2_3d}")

    dist = np.linalg.norm(p1_3d - p2_3d)
    print(f"\nEstimated distance: {dist:.4f} meters")

    if args.real_distance is not None:
        scale_ratio = args.real_distance / dist
        print(f"Real distance provided: {args.real_distance:.4f} meters")
        print(f"Scale ratio (real/estimated): {scale_ratio:.4f}")
        print(f"  -> if this is close to an integer or simple fraction, "
              f"there may be a unit conversion issue somewhere in the pipeline.")


if __name__ == "__main__":
    main()
