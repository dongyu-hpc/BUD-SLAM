# from vggt_slam.dynamic_mask import PersonMaskDetector
# import cv2

# detector = PersonMaskDetector()
# img = cv2.cvtColor(cv2.imread("/home/gdy/VGGT-SLAM/tesse_cd_apartment/frame00000.jpg"), cv2.COLOR_BGR2RGB)
# mask = detector.get_static_mask(img)  # True = keep, False = person/dynamic

# # 可视化确认mask对不对
# import numpy as np
# overlay = img.copy()
# overlay[~mask] = [255, 0, 0]  # 标红人的区域
# cv2.imwrite("mask_check.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
