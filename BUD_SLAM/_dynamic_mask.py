# # vggt_slam/dynamic_mask.py
# """
# Person detection/segmentation for filtering dynamic (human) regions out of
# the static SLAM pipeline. Uses YOLO-seg (person class only) since the
# dynamic content in this deployment is always people, occupying a small
# fraction (~10-15%) of the frame.
# """
# import numpy as np
# import torch
# from ultralytics import YOLO


# class PersonMaskDetector:
#     def __init__(self, model_name: str = "yolo11n-seg.pt", conf_threshold: float = 0.4, device: str = "cuda"):
#         """
#         model_name: any Ultralytics seg checkpoint pretrained on COCO (person is class 0).
#                     'n' (nano) is fastest; use 's'/'m' if you need better recall on small/occluded people.
#         conf_threshold: detection confidence threshold.
#         """
#         self.model = YOLO(model_name)
#         self.conf_threshold = conf_threshold
#         self.device = device
#         self.person_class_id = 0  # COCO class 0 = "person"

#     def get_static_mask(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
#         """
#         image_hwc_uint8: (H, W, 3) uint8 RGB image.
#         Returns a boolean mask of shape (H, W), True = static/background (safe to use for SLAM),
#         False = dynamic (person) region, to be excluded.
#         """
#         h, w = image_hwc_uint8.shape[:2]
#         results = self.model.predict(
#             image_hwc_uint8, conf=self.conf_threshold, classes=[self.person_class_id],
#             device=self.device, verbose=False,
#         )

#         static_mask = np.ones((h, w), dtype=bool)
#         r = results[0]
#         if r.masks is not None:
#             for mask_tensor in r.masks.data:  # each is (Hm, Wm) at model input resolution
#                 mask_np = mask_tensor.cpu().numpy()
#                 if mask_np.shape != (h, w):
#                     import cv2
#                     mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
#                 static_mask &= (mask_np < 0.5)

#         return static_mask

#     def get_static_masks_batch(self, images_hwc_uint8_list) -> np.ndarray:
#         """Convenience wrapper: list of (H,W,3) uint8 images -> (N,H,W) bool array."""
#         return np.stack([self.get_static_mask(img) for img in images_hwc_uint8_list], axis=0)
