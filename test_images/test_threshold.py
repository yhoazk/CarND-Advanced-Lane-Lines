#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# define color boundaries
boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128])
]


def get_yellow(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, (20,50,150),(40,255,255))
    return mask

def birds_eye_transform(img):
    # from the center of the image
    print(img.shape)
    h,w = img.shape[:2]
    #pts = np.array([[w*0.19, h*.95], [(w*0.47), h*0.62], [w*0.53, h*0.62], [w*0.83, h*.95]], np.int32)
    pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], np.int32)
    pts_rs = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts_rs],True,(0,255,255))
#   dst_pts =np.array([[(w / 4), 0],[(w / 4), h],[(w * 3 / 4), h],[(w * 3 / 4), 0]], dtype='float32')
    dst_pts =np.array([[320, 0],[320, 720],[970, 720],[960, 0]], dtype='float32')
    pts = np.float32(pts)
    mtx_bv = cv2.getPerspectiveTransform(pts, dst_pts)
    bird_img = cv2.warpPerspective(img, mtx_bv, (w,h) )
   # bird_img = bird_img[::-1]
    return img,  bird_img

def th_image(img):
    img_g = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    s_img = img_g[:,:,2]
    th_img = np.zeros_like(s_img)
    th_img[(s_img > 210)&(s_img<=255)] = 1
    edge = cv2.Canny(s_img,250,200)
    #sbl_img = np.abs(cv2.Sobel(th_img, cv2.CV_64F, 1,0))
    return edge

def get_hist_slice(img, slices=10):
    """
    Returns the possible location of the center of the line
    based on the pixels with val = 1
    """
    h_img = img.shape[0]
    w_img = int(img.shape[1]/2)
    location_l = []
    location_r = []
    location_ry = []
    location_ly = []
    location = {'l':location_l, 'r':location_r, 'ly':location_ly, 'ry':location_ry}
    for window in range(0,h_img, int(h_img/ slices)):
        sli = img[window:int(window+(h_img/slices)), :]
        sli_sum = np.sum(sli, axis=0)  # get the sum from all the columns
        sli_l, sli_r = (sli_sum[:w_img], sli_sum[w_img:])
        l_indx = np.mean(np.argpartition(sli_l, -5)[-5:])
        r_indx = np.mean(np.argpartition(sli_r, -5)[-5:]) + w_img

        location_ly.append(window)
        location_ry.append(window)
        location_l.append(l_indx)
        location_r.append(r_indx)
    return location

files = glob("./*.jpg")
for i in files:
    img = plt.imread(i)

    img = th_image(img)
    a,b = birds_eye_transform(img)
    lane_pts = get_hist_slice(b)
    #hist = np.sum(img[img.shape[0]/4:,:], axis=0)
    # b = th_image(b)
    ##### Find the lane fitting
    
    #####

    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(lane_pts['ly'], lane_pts['l'])
    ax2.imshow(b, cmap='gray')
    ax2.scatter(lane_pts['l'], lane_pts['ly'], s=50, c='red', marker='o')
    #plt.imshow(th_image(img))
    plt.show()
