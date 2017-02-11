import numpy as np
from  img_proc import *


class Line(img_proc):
    def __init__(self, side='l', debug=False):
        super().__init__()
        # was the line detected in the last iteration?
        self.detected = False
        # Number of frames to save
        self.FRAMES = 5
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients for the last fit
        self.last_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # a flag to indicate if mesasges and images are to be shown to debug
        self.debug = debug
        # determines if the lone is left line or right line_base_pos
        self.side = side
        # confidence
        self.confidence = 0
        # Error threshold
        self.poly = None
        self.last_poly = []

    def get_Curvature(self):
        pass

    def __get_ThresholdImg(self, img):
        img_g = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
        s_img = img_g[:,:,2]

        ret, th = cv2.threshold(s_img,127,255,cv2.THRESH_BINARY)
        #th_img = np.zeros_like(s_img)
        #th_img[(s_img > 50)] = 1
        #edge = cv2.Canny(s_img,250,200)
        #sbl_img = np.abs(cv2.Sobel(th_img, cv2.CV_64F, 0,1))
        return th

    def __remove_outliers(self, data_x, data_y, m=2):
        mean = np.mean(data_x)
        #print(mean)
        std_data = m*np.std(data_x)
        #print(std_data)
        #ret_data = [d for d in data if (abs(d-mean) < std_data) else mean]
        #ret_data = [d  if (abs(d-mean) < std_data) else mean for d in data]
        ret_data_y = []
        ret_data_x = []
        for x,y in zip(data_x, data_y):
            if (abs(x-mean) < std_data):
                ret_data_x.append(x)
                ret_data_y.append(y)

        return (ret_data_y, ret_data_x)

    def __get_hist_slice(self, img, slices=10, margin=100):
        """
        Returns the possible location of the center of the line
        based on the pixels with val = 1
        The image received has to be binarized.
        """
        h_img = img.shape[0]
        w_img = img.shape[1]
        location_l = []
        location_r = []
        location_ry = []
        location_ly = []

        """
        Createa a mask to ignore the center and the extremes of the image.
        ****-----***-----****
        ****-----***-----****
        ****-----***-----****
        ****-----***-----****
        """
        zero_patch = np.zeros((h_img, margin))
        one_patch  = np.ones((h_img, (w_img//2)-(1.5*margin)))

        mask = np.c_[zero_patch, one_patch]
        mask = np.c_[mask, zero_patch]
        mask = np.c_[mask, one_patch]
        mask = np.c_[mask, zero_patch]
        img = np.uint8(img)
        mask = np.uint8(mask)
        # apply mask to entire image
        img = cv2.bitwise_and(img,img,mask = mask)
        if self.debug:
            plt.imshow(img)
            plt.show()

        for window in reversed(range(0,h_img, int(h_img/ slices))):
            sli = img[window:int(window+(h_img/slices)), :]
            sli_sum = np.sum(sli, axis=0)  # get the sum from all the columns
            """
            Add a margin to the histogram to not take pixels at the far left or right
            """
            sli_l, sli_r = (sli_sum[:w_img//2], sli_sum[w_img//2:])

            # get the location of 5 top max elements
            l_arg = np.argpartition(sli_l, -5)[-5:]
            r_arg = np.argpartition(sli_r, -5)[-5:]
            # Get the value of the max values and decide of this portion
            # of frame contains something interesiting
            mag_r = sum(sli_r[r_arg])
            mag_l = sum(sli_l[l_arg])
            if mag_l > 100:
                l_indx = np.mean(l_arg)
                location_l.append(l_indx)
                location_ly.append(window)


            if mag_r > 100:
                r_indx = np.mean(r_arg) + w_img//2
                location_ry.append(window)
                location_r.append(r_indx)

            #print("r_indx: " + str(r_indx) + " sli_r: " + str(sli_r))
            # add condtion for the case when the index is 0
        # if a point is 0 make its value the median btw the point before and the point after
        location = {'l':location_l, 'r':location_r, 'ly':location_ly, 'ry':location_ry}
        if self.debug == True:
            print("l : " + str(len(location_l)))
            print(location_l)
            print("ly : " + str(len(location_ly)))
            print(location_ly)

            print("r : " + str(len(location_r)))
            print(location_r)
            print("ry : " + str(len(location_ry)))
            print(location_ry)

        # add the located points in the array
        if len(self.recent_xfitted) >= self.FRAMES:
            self.recent_xfitted.pop(0) # remove the oldest element
        # add the newest element to the queue
        self.recent_xfitted.append(location[self.side]) # add the newst values of x

        # temp
#        if side == 'l':
        return location

    def __calc_poly():
        pass

    def update(self, img):
        th_img = self.__get_ThresholdImg(img)
        b_img  = self.get_birdsView(th_img)
        lane_pts = self.__get_hist_slice(b_img)

        if len(lane_pts[self.side]) > 2:
            # Find the polynomial
            try:
                fit, v  = np.polyfit(*self.__remove_outliers(lane_pts[self.side], lane_pts[self.side+'y']), deg=2, cov=True)
            except:
                fit = np.polyfit(*self.__remove_outliers(lane_pts[self.side], lane_pts[self.side+'y']), deg=2)
                v = None
            self.last_fit = self.current_fit
            self.current_fit = fit
        else:
            # use the past fit as we do not have enough information to decide
            self.current_fit = self.last_fit
            fit = self.last_fit
            v = None
        # if v == None then there was an exception in fittig the polynomial
        if v == None:
            # The covariance matrix could not be obtained, compare with previous
            # fit
            self.poly = np.poly1d(fit)
        else:
            # v indicates the error in fitting hte line, if its big
            # the confidence drops
            error_r = np.sum(np.abs(v[:][:][2]))
            # decide if add or not te new values
            # TODO add filter here
            # if error is small add it as best fit
            self.poly = np.poly1d(fit)
            if len(self.last_poly) > self.FRAMES:
                self.last_poly.pop(0)

            self.last_poly.append(self.poly)
        return b_img

    def get_LinePoly(self):
        """
        This returns the filtered polynomial
        To be used in the lane class
        """
        if len(self.last_poly) <= 2:
            # the first element
            self.best_fit = self.poly
        else:
            cofs = [x.coeffs for x in self.last_poly]
            cofs = np.mean(cofs, axis=0)


            self.best_fit = np.poly1d(cofs)
        return self.best_fit
