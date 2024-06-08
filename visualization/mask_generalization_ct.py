# the code for generating masks for ABD-CT  

import os
import cv2
import numpy as np
import SimpleITK as itk
import matplotlib.pyplot as plt
import matplotlib
import cv2
import SimpleITK as itk

# Abd_CT dataset

abd_ct_pth = './data/Abd_CT/'
abd_mri_pth = './data/Abd_MRI/'
cmr_pth = './data/CMR/'

abd_ct_gt_pth = os.path.join(abd_ct_pth, 'Abd_CT_x_GT.nii.gz')              # the ground truth mask, x is case number 
abd_ct_img_pth = os.path.join(abd_ct_pth, 'Abd_CT_x.nii.gz')                # the image 
abd_ct_liver_pth = os.path.join(abd_ct_pth, 'prediction_x_LIVER.nii.gz')    # the liver prediction of case x 
abd_ct_spleen_pth = os.path.join(abd_ct_pth, 'prediction_x_SPLEEN.nii.gz')  # the spleen prediction of case x
abd_ct_rk_pth = os.path.join(abd_ct_pth, 'prediction_x_RK.nii.gz')          # the right kidney prediction of case x
abd_ct_lk_pth = os.path.join(abd_ct_pth, 'prediction_x_LK.nii.gz')          # the left kidney prediction of case x


abd_ct_gt = itk.GetArrayFromImage(itk.ReadImage(abd_ct_gt_pth))  # (100, 257, 257)
abd_ct_img = itk.GetArrayFromImage(itk.ReadImage(abd_ct_img_pth))  # (100, 257, 257)

abd_ct_gt[abd_ct_gt == 200] = 1
abd_ct_gt[abd_ct_gt == 500] = 2
abd_ct_gt[abd_ct_gt == 600] = 3

# ********************************liver**********************************
abd_ct_liver = itk.GetArrayFromImage(itk.ReadImage(abd_ct_liver_pth))  
abd_ct_liver_gt = 1 * (abd_ct_gt == 6)
idx = abd_ct_liver_gt.sum(axis=(1, 2)) > 0
abd_ct_liver_gt = abd_ct_liver_gt[idx]
abd_ct_img_liver = abd_ct_img[idx]

abd_ct_liver_gt_show = abd_ct_liver_gt[16] * 200  # choose the 16th slice of case x to illustrate, you also can choose other slices
abd_ct_img_show = abd_ct_img_liver[16]
abd_ct_liver_show = abd_ct_liver[16] * 200

abd_ct_liver_spt = abd_ct_liver_gt[13] * 200   # choose the 13th slice of case x as support image.  
abd_ct_img_spt = abd_ct_img_liver[13]

cv2.imwrite("./data/Abd_CT/abd_ct_liver_gt.png", abd_ct_liver_gt_show)    # ground truth 
cv2.imwrite("./data/Abd_CT/abd_ct_liver_img.png", abd_ct_img_show)        # 
cv2.imwrite("./data/Abd_CT/abd_ct_liver.png", abd_ct_liver_show)

cv2.imwrite("./data/Abd_CT/abd_ct_liver_spt.png", abd_ct_liver_spt)
cv2.imwrite("./data/Abd_CT/abd_ct_liver_img_spt.png", abd_ct_img_spt)

# **********************************spleen*********************************
abd_ct_spleen = itk.GetArrayFromImage(itk.ReadImage(abd_ct_spleen_pth))
abd_ct_spleen_gt = 1 * (abd_ct_gt == 1)
idx = abd_ct_spleen_gt.sum(axis=(1, 2)) > 0
abd_ct_spleen_gt = abd_ct_spleen_gt[idx]
abd_ct_img_spleen = abd_ct_img[idx]

abd_ct_spleen_gt_show = abd_ct_spleen_gt[14]*200
abd_ct_img_show = abd_ct_img_spleen[14]

abd_ct_spleen_spt = abd_ct_spleen_gt[8]*200
abd_ct_img_spt = abd_ct_img_spleen[8]

cv2.imwrite("./data/Abd_CT/abd_ct_spleen_gt.png", abd_ct_spleen_gt_show)
cv2.imwrite("./data/Abd_CT/abd_ct_spleen_img.png", abd_ct_img_show)
cv2.imwrite("./data/Abd_CT/abd_ct_spleen.png", abd_ct_spleen_show)

cv2.imwrite("./data/Abd_CT/abd_ct_spleen_spt.png", abd_ct_spleen_spt)
cv2.imwrite("./data/Abd_CT/abd_ct_spleen_img_spt.png", abd_ct_img_spt)

# **********************************RK************************************
abd_ct_rk = itk.GetArrayFromImage(itk.ReadImage(abd_ct_rk_pth))
abd_ct_rk_gt = 1 * (abd_ct_gt == 2)
idx = abd_ct_rk_gt.sum(axis=(1, 2)) > 0
abd_ct_rk_gt = abd_ct_rk_gt[idx]
abd_ct_img_rk = abd_ct_img[idx]

abd_ct_rk_gt_show = abd_ct_rk_gt[18]*200
abd_ct_img_show = abd_ct_img_rk[18]

abd_ct_rk_spt = abd_ct_rk_gt[10]*200
abd_ct_img_spt = abd_ct_img_rk[10]

cv2.imwrite("./data/Abd_CT/abd_ct_rk_gt.png", abd_ct_rk_gt_show)
cv2.imwrite("./data/Abd_CT/abd_ct_rk_img.png", abd_ct_img_show)
cv2.imwrite("./data/Abd_CT/abd_ct_rk.png", abd_ct_rk_show)

cv2.imwrite("./data/Abd_CT/abd_ct_rk_spt.png", abd_ct_rk_spt)
cv2.imwrite("./data/Abd_CT/abd_ct_rk_img_spt.png", abd_ct_img_spt)

# *********************************LK**************************************
abd_ct_lk = itk.GetArrayFromImage(itk.ReadImage(abd_ct_lk_pth))
abd_ct_lk_gt = 1 * (abd_ct_gt == 3)
idx = abd_ct_lk_gt.sum(axis=(1, 2)) > 0
abd_ct_lk_gt = abd_ct_lk_gt[idx]
abd_ct_img_lk = abd_ct_img[idx]

abd_ct_lk_gt_show = abd_ct_lk_gt[17]*200
abd_ct_img_show = abd_ct_img_lk[17]

abd_ct_lk_spt = abd_ct_lk_gt[8]*200
abd_ct_img_spt = abd_ct_img_lk[8]

cv2.imwrite("./data/Abd_CT/abd_ct_lk_gt.png", abd_ct_lk_gt_show)
cv2.imwrite("./data/Abd_CT/abd_ct_lk_img.png", abd_ct_img_show)
cv2.imwrite("./data/Abd_CT/abd_ct_lk.png", abd_ct_lk_show)

cv2.imwrite("./data/Abd_CT/abd_ct_lk_spt.png", abd_ct_lk_spt)
cv2.imwrite("./data/Abd_CT/abd_ct_lk_img_spt.png", abd_ct_img_spt)


