# the code for generating masks for ABD-MRI

import os
import cv2
import numpy as np
import SimpleITK as itk
import matplotlib.pyplot as plt
import matplotlib
import cv2
import SimpleITK as itk

# Abd_MRI dataset


abd_ct_pth = './data/Abd_CT/'
abd_mri_pth = './data/Abd_MRI/'
cmr_pth = './data/CMR/'

abd_mri_gt_pth = os.path.join(abd_mri_pth, 'Abd_MRI_x_GT.nii.gz')                 # the ground truth mask, x is case number  
abd_mri_img_pth = os.path.join(abd_mri_pth, 'Abd_MRI_x.nii.gz')                   # the image 

abd_mri_liver_pth = os.path.join(abd_mri_pth, 'prediction_x_LIVER.nii.gz')        # the liver prediction of case x 
abd_mri_spleen_pth = os.path.join(abd_mri_pth, 'prediction_x_SPLEEN.nii.gz')      # the spleen prediction of case x 
abd_mri_rk_pth = os.path.join(abd_mri_pth, 'prediction_x_RK.nii.gz')              # the right kidney prediction of case x 
abd_mri_lk_pth = os.path.join(abd_mri_pth, 'prediction_x_LK.nii.gz')              # the left kidney prediction of case x 

abd_mri_gt = itk.GetArrayFromImage(itk.ReadImage(abd_mri_gt_pth))  
abd_mri_img = itk.GetArrayFromImage(itk.ReadImage(abd_mri_img_pth))  


abd_mri_gt[abd_mri_gt == 200] = 1
abd_mri_gt[abd_mri_gt == 500] = 2
abd_mri_gt[abd_mri_gt == 600] = 3

# ********************************liver**********************************
abd_mri_liver = itk.GetArrayFromImage(itk.ReadImage(abd_mri_liver_pth))
abd_mri_liver_gt = 1 * (abd_mri_gt == 1)
idx = abd_mri_liver_gt.sum(axis=(1, 2)) > 0
abd_mri_liver_gt = abd_mri_liver_gt[idx]
abd_mri_img_liver = abd_mri_img[idx]

abd_mri_liver_gt_show = abd_mri_liver_gt[11] * 200    # choose the 11th slice of case x to illustrate, you also can choose other slices
abd_mri_img_show_1 = abd_mri_img_liver[11] / 4.5
abd_mri_liver_show = abd_mri_liver[11] * 200

abd_mri_liver_spt = abd_mri_liver_gt[5] * 200    # choose the 5th slice of case x as support image.
abd_mri_img_spt = abd_mri_img_liver[5] / 4.5

cv2.imwrite("./data/Abd_MRI/abd_mri_liver_gt.png", abd_mri_liver_gt_show)   # the ground truth 
cv2.imwrite("./data/Abd_MRI/abd_mri_liver_img.png", abd_mri_img_show_1)       # the image
cv2.imwrite("./data/Abd_MRI/abd_mri_liver.png", abd_mri_liver_show)         # the liver prediction 

cv2.imwrite("./data/Abd_MRI/abd_mri_liver_spt.png", abd_mri_liver_spt)      # the liver mask of support image 
cv2.imwrite("./data/Abd_MRI/abd_mri_liver_img_spt.png", abd_mri_img_spt)    # the support image 

# **********************************spleen*********************************
abd_mri_spleen = itk.GetArrayFromImage(itk.ReadImage(abd_mri_spleen_pth))
abd_mri_spleen_gt = 1 * (abd_mri_gt == 4)
idx = abd_mri_spleen_gt.sum(axis=(1, 2)) > 0
abd_mri_spleen_gt = abd_mri_spleen_gt[idx]
abd_mri_img_spleen = abd_mri_img[idx]

abd_mri_spleen_gt_show = abd_mri_spleen_gt[10]*200
abd_mri_img_show_2 = abd_mri_img_spleen[10] / 4.5

abd_mri_spleen_spt = abd_mri_spleen_gt[4]*200
abd_mri_img_spt = abd_mri_img_spleen[4] / 4.5

cv2.imwrite("./data/Abd_MRI/abd_mri_spleen_gt.png", abd_mri_spleen_gt_show)
cv2.imwrite("./data/Abd_MRI/abd_mri_spleen_img.png", abd_mri_img_show_2)
cv2.imwrite("./data/Abd_MRI/abd_mri_spleen.png", abd_mri_spleen_show)

cv2.imwrite("./data/Abd_MRI/abd_mri_spleen_spt.png", abd_mri_spleen_spt)
cv2.imwrite("./data/Abd_MRI/abd_mri_spleen_img_spt.png", abd_mri_img_spt)

# **********************************RK************************************
abd_mri_rk = itk.GetArrayFromImage(itk.ReadImage(abd_mri_rk_pth))
abd_mri_rk_gt = 1 * (abd_mri_gt == 2)
idx = abd_mri_rk_gt.sum(axis=(1, 2)) > 0
abd_mri_rk_gt = abd_mri_rk_gt[idx]
abd_mri_img_rk = abd_mri_img[idx]

abd_mri_rk_gt_show = abd_mri_rk_gt[10]*200
abd_mri_img_show_3 = abd_mri_img_rk[10] / 4.5

abd_mri_rk_spt = abd_mri_rk_gt[11]*200
abd_mri_img_spt = abd_mri_img_rk[11] / 4.5

cv2.imwrite("./data/Abd_MRI/abd_mri_rk_gt.png", abd_mri_rk_gt_show)
cv2.imwrite("./data/Abd_MRI/abd_mri_rk_img.png", abd_mri_img_show_3)
cv2.imwrite("./data/Abd_MRI/abd_mri_rk.png", abd_mri_rk_show)

cv2.imwrite("./data/Abd_MRI/abd_mri_rk_spt.png", abd_mri_rk_spt)
cv2.imwrite("./data/Abd_MRI/abd_mri_rk_img_spt.png", abd_mri_img_spt)

# *********************************LK**************************************
abd_mri_lk = itk.GetArrayFromImage(itk.ReadImage(abd_mri_lk_pth))
abd_mri_lk_gt = 1 * (abd_mri_gt == 3)
idx = abd_mri_lk_gt.sum(axis=(1, 2)) > 0
abd_mri_lk_gt = abd_mri_lk_gt[idx]
abd_mri_img_lk = abd_mri_img[idx]

abd_mri_lk_gt_show = abd_mri_lk_gt[11]*200
abd_mri_img_show_4 = abd_mri_img_lk[11] / 4.5

abd_mri_lk_spt = abd_mri_lk_gt[5]*200
abd_mri_img_spt = abd_mri_img_lk[5] / 4.5

cv2.imwrite("./data/Abd_MRI/abd_mri_lk_gt.png", abd_mri_lk_gt_show)
cv2.imwrite("./data/Abd_MRI/abd_mri_lk_img.png", abd_mri_img_show_4)
cv2.imwrite("./data/Abd_MRI/abd_mri_lk.png", abd_mri_lk_show)

cv2.imwrite("./data/Abd_MRI/abd_mri_lk_spt.png", abd_mri_lk_spt)
cv2.imwrite("./data/Abd_MRI/abd_mri_lk_img_spt.png", abd_mri_img_spt)

