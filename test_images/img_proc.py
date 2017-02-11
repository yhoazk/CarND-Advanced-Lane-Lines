import cv2
import matplotlib.pyplot as plt
import numpy as np


class img_proc():
    def __init__(self):
        # source points for birds eye transformation
        self.src_pts = None
        # destination points for birds eye transformation
        self.dst_pts = None
        # The distorion coefficients
        self.dist_cof = None

    def camera_calibration(self, regexp):
        """
        regexp is the regular expression for the glob
        """
        pass

    def test_in(self):
        print("inheritance working")
