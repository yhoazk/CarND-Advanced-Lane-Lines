import cv2
import numpy as np
import matplotlib.pyplot  as plt
#from  img_proc import *
"""
This class contains two lanes and compares both lines for coherency and
correctess.

"""
IMAGE_WIDTH = 1280

class Lane():
    def __init__(self, inLine_l, inLine_r, ls_length=20):
        self.line_l = inLine_l
        self.line_r = inLine_r
        # the curvature estimated with both lines
        self.radio = None
        # offset from the center
        self.lane_offset = None
        self.lane_vertices = None
        self.center_poly = [np.array([False])]
        self.line_segment_length = ls_length

    def check_curvature(self, fit_l, fit_r):
        """
        :param fit_l: polynomial for the left lane
        :param fit_r: polynomial for the right lane
        :return:
         This function compares the curvature of each line
        """
        pass

    def check_separation(self, fit_l, fit_r):
        """
        :param fit_l:
        :param fit_r:
        :return:
        """
        pass

    def check_Lines(self):
        """
        Checks if the lines are more or less the same curve.
        if they are more or less parallel.
        """

        pass

    def get_laneOffset(self):
        return self.lane_offset
    def process_laneOffset(self):
        """
        Calculate the offset from the center of the lane
        Finds the center polinomial based on the left and right polinomials and
        then substracts from the supposed center, ie 1280/2.
        """
        center_line = np.poly1d(np.mean([self.line_l.get_LinePoly().coeffs, self.line_r.get_LinePoly().coeffs], axis=0))
        # store the center line polynomial
        self.center_poly = center_line
        center_point = IMAGE_WIDTH/2 - center_line(715)
        offset_from_center =center_point* self.line_l.x_pxm
        self.lane_offset = offset_from_center
        return center_point


    def get_Curvature(self):
        """
        Calculate the offset from the center of the lane
        """
        return str(0.5*(self.line_l.get_CurveRad() + self.line_r.get_CurveRad()))

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
        self.process_laneOffset()
        shade = np.uint8(np.zeros((720,1280)))
        shade = cv2.fillConvexPoly(shade,lane_vertices,1)
        # add calculated lines to the image


        for n in np.linspace(0,720-self.line_segment_length, 720//self.line_segment_length, dtype=int):
            p1 = (int(self.center_poly(n)), n)
            p2 = (int(self.center_poly(n+self.line_segment_length)), n+self.line_segment_length)
            shade = cv2.line(shade, p1, p2, color=(255,255,255), thickness=5, lineType=4)
        # shows the shade calculates
        #plt.imshow(shade, cmap='gray')
        #plt.show()
        # reverse birds view
        warp  =  self.line_l.get_reverseBirdsView(shade)
        warp_3 = np.dstack([120*warp,150*warp,20*warp])
        shaded_lane= cv2.addWeighted(img, 0.8, warp_3, 0.4,0)
        # add the curvature and offset
        processed_img = cv2.putText(shaded_lane,
                                    "Curvature: " + self.get_Curvature(),
                                    (500,50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .50,
                                    color=(255,120,40))
        processed_img = cv2.putText(shaded_lane,
                                    "Deviation from center: " + str(self.get_laneOffset()),
                                        (500,90),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .50,
                                    color=(0,255,0))

        return processed_img



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
