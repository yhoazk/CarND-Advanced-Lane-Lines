import cv2
import matplotlib.pyplot as plt
import numpy as np


class img_proc():
    def __init__(self):
        print("img proc called")
        # image shape
        self.w_img = 1280
        self.h_img = 720
        # source points for birds eye transformation
        self.src_pts = np.array([[320, 0],[320, 720],[970, 720],[960, 0]], dtype='float32')
        # destination points for birds eye transformation
        self.dst_pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], dtype='float32').reshape((-1,1,2))
        # birds eye matrix
        self.bv_matrix = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        # reversed birds eye matrix
        self.rev_bv_matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        # The distorion coefficients
        self.dist_cof = None
        print("img proc done")

    def camera_calibration(self, regexp):
        """
        regexp is the regular expression for the glob
        """
        pass

    def get_birdsView(self, img):
        return cv2.warpPerspective(img, self.bv_matrix, (self.w_img,self.h_img) )

    def get_reverseBirdsView(self, img):
        return cv2.warpPerspective(img, self.rev_bv_matrix, (self.w_img,self.h_img) )

    def test_in(self):
        print("inheritance working")
