#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np


def birds_eye_transform(img):
    # from the center of the image
    print(img.shape)
    h,w = img.shape[:2]
    pts = np.array([[w*0.19, h*.95], [(w*0.47), h*0.62], [w*0.53, h*0.62], [w*0.83, h*.95]], np.int32)
    pts_rs = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts_rs],True,(0,255,255))
    dst_pts =np.array([[(w / 4), 0],[(w / 4), h],[(w * 3 / 4), h],[(w * 3 / 4), 0]], dtype='float32')
#    dst_pts =np.array([[320, 0],[320, 720],[970, 720],[960, 0]], dtype='float32')
    pts = np.float32(pts)
    mtx_bv = cv2.getPerspectiveTransform(pts, dst_pts)
    bird_img = cv2.warpPerspective(img, mtx_bv, (w,h) )
    bird_img = bird_img[::-1]
    return img,  bird_img

def th_image(img):
    img_g = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    s_img = img_g[:,:,2]
    th_img = np.zeros_like(s_img)
    th_img[(s_img > 200)&(s_img<=255)] = 1
    sbl_img = np.abs(cv2.Sobel(th_img, cv2.CV_64F, 1,0))
    return sbl_img

files = glob("./*.jpg")
for i in files:
    img = plt.imread(i)

    a,b = birds_eye_transform(img)
#    a = th_image(a)
#    b = th_image(b)
    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(a)
    ax2.imshow(b)
    plt.show()
