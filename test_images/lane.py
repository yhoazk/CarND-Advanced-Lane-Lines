

"""
This class contains two lanes and compares both lines for coherency and
correctess.

"""

class lane():
    def __init__(self, inLine_l, inLine_r):
        self.line_l = inLine_l
        self.line_r = inLine_r
        # the curvature estimated with both lines
        self.radio = None
        # offset from the center
        self.lane_offset = None

    def check_Lines():
        """
        Checks if the lines are more or less the same curve.
        if they are more or less parallel.
        """
        pass

    def get_laneOffset():
        """
        Calculate the offset from the center of the lane
        """
        pass

    def get_lane_area(poly_l, poly_r, n=10):
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

        return np.append(np.c_[fy_l,x], np.c_[fy_r,x][::-1], axis=0)
