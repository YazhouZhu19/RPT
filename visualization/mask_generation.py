# code for mask generation 
import os
import cv2
import numpy as np
import SimpleITK as itk
import matplotlib.pyplot as plt
import matplotlib
import cv2
import SimpleITK as itk

# CMR dataset

cmr_pth = './data/CMR/'

cmr_gt_pth = os.path.join(cmr_pth, 'CMR_13_GT.nii.gz')
cmr_img_pth = os.path.join(cmr_pth, 'CMR_13.nii.gz')
cmr_lvbp_pth = os.path.join(cmr_pth, 'prediction_13_LV-BP.nii.gz')
cmr_lvmyo_pth = os.path.join(cmr_pth, 'prediction_13_LV-MYO.nii.gz')
cmr_rv_pth = os.path.join(cmr_pth, 'prediction_13_RV.nii.gz')

cmr_gt = itk.GetArrayFromImage(itk.ReadImage(cmr_gt_pth))  # (100, 257, 257)
cmr_img = itk.GetArrayFromImage(itk.ReadImage(cmr_img_pth))  # (100, 257, 257)

cmr_gt[cmr_gt == 200] = 1
cmr_gt[cmr_gt == 500] = 2
cmr_gt[cmr_gt == 600] = 3

# ********************************lvbp**********************************
cmr_lvbp = itk.GetArrayFromImage(itk.ReadImage(cmr_lvbp_pth))
cmr_lvbp_gt = 1 * (cmr_gt == 2)
idx = cmr_lvbp_gt.sum(axis=(1, 2)) > 0
cmr_lvbp_gt = cmr_lvbp_gt[idx]
cmr_img_lvbp = cmr_img[idx]

cmr_lvbp_gt_show = cmr_lvbp_gt[4]*200
cmr_img_show = cmr_img_lvbp[4] / 4.5

cmr_lvbp_spt = cmr_lvbp_gt[2]*200
cmr_img_spt = cmr_img_lvbp[2] / 4.5

cv2.imwrite("./data/CMR/cmr_lvbp_gt.png", cmr_lvbp_gt_show)
cv2.imwrite("./data/CMR/cmr_lvbp_img.png", cmr_img_show)
cv2.imwrite("./data/CMR/cmr_lvbp.png", cmr_img_lvbp_show)

cv2.imwrite("./data/CMR/cmr_lvbp_spt.png", cmr_lvbp_spt)
cv2.imwrite("./data/CMR/cmr_lvbp_img_spt.png", cmr_img_spt)

# **********************************lvmyo*********************************
cmr_lvmyo = itk.GetArrayFromImage(itk.ReadImage(cmr_lvmyo_pth))
cmr_lvmyo_gt = 1 * (cmr_gt == 1)
idx = cmr_lvmyo_gt.sum(axis=(1, 2)) > 0
cmr_lvmyo_gt = cmr_lvmyo_gt[idx]
cmr_img_lvmyo = cmr_img[idx]

cmr_lvmyo_gt_show = cmr_lvmyo_gt[6]*200
cmr_img_show = cmr_img_lvmyo[6] / 4.5

cmr_lvmyo_spt = cmr_lvmyo_gt[3]*200
cmr_img_spt = cmr_img_lvmyo[3] / 4.5

cv2.imwrite("./data/CMR/cmr_lvmyo_gt.png", cmr_lvmyo_gt_show)
cv2.imwrite("./data/CMR/cmr_lvmyo_img.png", cmr_img_show)
cv2.imwrite("./data/CMR/cmr_lvmyo.png", cmr_img_lvmyo_show)

cv2.imwrite("./data/CMR/cmr_lvmyo_spt.png", cmr_lvmyo_spt)
cv2.imwrite("./data/CMR/cmr_lvmyo_img_spt.png", cmr_img_spt)

# **********************************rv************************************
cmr_rv = itk.GetArrayFromImage(itk.ReadImage(cmr_rv_pth))
cmr_rv_gt = 1 * (cmr_gt == 3)
idx = cmr_rv_gt.sum(axis=(1, 2)) > 0
cmr_rv_gt = cmr_rv_gt[idx]
cmr_img_rv = cmr_img[idx]

cmr_rv_gt_show = cmr_rv_gt[2]*200
cmr_img_show = cmr_img_rv[2] / 4.5


cmr_rv_spt = cmr_rv_gt[5]*200
cmr_img_spt = cmr_img_rv[5] / 4.5

cv2.imwrite("./data/CMR/cmr_rv_gt.png", cmr_rv_gt_show)
cv2.imwrite("./data/CMR/cmr_rv_img.png", cmr_img_show)
cv2.imwrite("./data/CMR/cmr_rv.png", cmr_img_rv_show)

cv2.imwrite("./data/CMR/cmr_rv_spt.png", cmr_rv_spt)
cv2.imwrite("./data/CMR/cmr_rv_img_spt.png", cmr_img_spt)

