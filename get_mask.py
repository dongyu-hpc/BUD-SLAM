"""
用法:
    python get_mask.py --frame_dir ./frames --out_dir ./masks --save_tensor dyn_masks.pt

    python get_mask.py --frame_dir /home/gdy/VGGT-SLAM/tesse_cd_apartment --out_dir ./masks --save_tensor dyn_masks.pt

"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

_dino_processor = None
_dino_model = None
_sam_processor = None
_sam_model = None


def _load_models(device="cuda",
                  dino_ckpt="IDEA-Research/grounding-dino-tiny",
                  sam_ckpt="facebook/sam-vit-base"):
    global _dino_processor, _dino_model, _sam_processor, _sam_model
    if _dino_model is None:
        _dino_processor = AutoProcessor.from_pretrained(dino_ckpt)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_ckpt).to(device).eval()
    if _sam_model is None:
        _sam_processor = SamProcessor.from_pretrained(sam_ckpt)
        _sam_model = SamModel.from_pretrained(sam_ckpt).to(device).eval()


@torch.no_grad()
@torch.no_grad()
def get_mask(image, text="person.", device="cuda", box_threshold=0.3, text_threshold=0.25):
    """
    image: HxWx3 numpy array (RGB, uint8) 或 PIL.Image
    返回: HxW numpy array, 0/1, 1表示人（动态区域）
    """
    _load_models(device=device)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    W, H = image.size

    inputs = _dino_processor(images=image, text=text, return_tensors="pt").to(device)
    outputs = _dino_model(**inputs)
    results = _dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)]
    )[0]

    boxes = results["boxes"].cpu().numpy()
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(boxes) == 0:
        return mask

    sam_inputs = _sam_processor(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    sam_outputs = _sam_model(**sam_inputs)
    masks = _sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu()
    )[0].numpy()

    if masks.ndim == 4:
        masks = masks[:, 0]

    for m in masks:
        mask = np.maximum(mask, (m > 0).astype(np.uint8))

    return mask



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.jpg")
    parser.add_argument("--out_dir", type=str, default="masks")
    parser.add_argument("--save_tensor", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=100, help="每隔N帧保存一次中间结果")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frame_dir, args.pattern)))
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {args.frame_dir} with pattern {args.pattern}")
    if args.max_frames is not None:
        frame_paths = frame_paths[:args.max_frames]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Found {len(frame_paths)} frames", flush=True)

    masks = []
    import time
    t_start = time.time()
    for i, path in enumerate(frame_paths):
        img = np.array(Image.open(path).convert("RGB"))
        mask = get_mask(img, device=args.device)
        masks.append(mask)

        out_name = os.path.splitext(os.path.basename(path))[0] + "_mask.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out_dir, out_name))

        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        eta = avg * (len(frame_paths) - i - 1)
        print(f"[{i+1}/{len(frame_paths)}] {os.path.basename(path)} "
              f"dyn_ratio={mask.mean():.3f} avg={avg:.2f}s/frame ETA={eta/60:.1f}min", flush=True)

        # 定期保存中间结果，防止中断丢失
        if args.save_tensor and (i + 1) % args.checkpoint_every == 0:
            tmp = torch.from_numpy(np.stack(masks)).float()
            torch.save(tmp, args.save_tensor + f".partial_{i+1}.pt")

    if args.save_tensor is not None:
        dyn_masks = torch.from_numpy(np.stack(masks)).float()  # [S, H, W]
        torch.save(dyn_masks, args.save_tensor)
        print(f"Saved sequence tensor {dyn_masks.shape} to {args.save_tensor}", flush=True)



if __name__ == "__main__":
    main()
