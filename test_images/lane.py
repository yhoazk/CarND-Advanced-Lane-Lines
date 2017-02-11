import cv2
import numpy as np
import matplotlib.pyplot  as plt
#from  img_proc import *
"""
This class contains two lanes and compares both lines for coherency and
correctess.

"""

class Lane():
    def __init__(self, inLine_l, inLine_r):
        self.line_l = inLine_l
        self.line_r = inLine_r
        # the curvature estimated with both lines
        self.radio = None
        # offset from the center
        self.lane_offset = None
        self.lane_vertices = None

    def check_Lines(self):
        """
        Checks if the lines are more or less the same curve.
        if they are more or less parallel.
        """
        pass

    def get_laneOffset(self):
        """
        Calculate the offset from the center of the lane
        """
        pass

    def process_lane(self, img):
        """
        :img: the image of hte current frame where the shade will be added
        Complete processing of the lane
        - Get filtered polynomial x&y
        - Fill lane area
        - Calculate lane offset
        - Calculate curve radius_of_curvature for lane
        - Print them in the image
        """
        smoth_poly_l = self.line_l.get_LinePoly()
        smoth_poly_r = self.line_r.get_LinePoly()
        lane_vertices = self.__get_lane_area(smoth_poly_l, smoth_poly_r)
        shade = np.uint8(np.zeros((720,1280)))
        shade = cv2.fillConvexPoly(shade,lane_vertices,1)
        plt.imshow(shade, cmap='gray')
        plt.show()
        # reverse birds view
        warp  =  self.line_l.get_reverseBirdsView(shade)
        warp_3 = np.dstack([120*warp,150*warp,20*warp])
        shaded_lane= cv2.addWeighted(img, 0.6, warp_3, 0.4,0)
        return shaded_lane



    def __get_lane_area(self, poly_l, poly_r, n=10):
        """
        Create a list of points that describe the
        area between the lanes, starts from the bottom
        left corner
        poly_l is a 1d polynomial
        poly_r is a 1d polynomial
        """
        # set the list of points as needed by polyfill
        x = np.linspace(0, 720, n)
        fy_l = poly_l(x)
        fy_r = poly_r(x)

        return np.int32(np.append(np.c_[fy_l,x], np.c_[fy_r,x][::-1], axis=0))
