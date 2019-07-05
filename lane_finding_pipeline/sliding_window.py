import numpy as np
import cv2
from lane_finding_pipeline.piplineinterface import PipeLineInterface

class LaneFinding(PipeLineInterface):
    def __init__(self, nwindow=9, nmargin=100, minpix=50):
        self.nwindows = nwindow
        self.margin = nmargin
        self.minpix = minpix
        self.left_fit = None
        self.right_fit = None
        self.curve_msg, self.offset_msg = "", ""

    def findHistogram(self, image):
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return histogram, leftx_base, rightx_base

    def searchAroundPoly(self, image):
        out_img = np.dstack((image, image, image))

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - self.margin)) &
                          (nonzerox < (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + self.margin)))

        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - self.margin)) &
                           (nonzerox < (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img


    def fitPoly(self, img_shape, leftx, lefty, rightx, righty):
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        try:
            left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
            right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return left_fitx, right_fitx, ploty

    def slidingWindowFindLanePixel(self, image):
        out_img = np.dstack((image, image, image))

        histogram, leftx_base, rightx_base = self.findHistogram(image)

        window_height = np.int(image.shape[0] // self.nwindows)
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw the windows on the visualization image not needed in the final pipeline
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

           # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds])) 

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img
    
    def getAverageCurvature(self):
        return self.curve_msg
    
    def getOffsetMsg(self):
        return self.offset_msg
    
    def calculateMetaData(self, img_size, left_x_predictions, right_x_predictions):
        num_rows, num_cols = self.img_size[0], self.img_size[1]
        def measure_radius_of_curvature(x_values):
            ym_per_pix = 30/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meters per pixel in x dimension
            # If no pixels were found return None
            y_points = np.linspace(0, num_rows-1, num_rows)
            y_eval = np.max(y_points)

            # Fit new polynomials to x,y in world space
            fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
            curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
            return curverad
        
        if self.left_fit is None or self.right_fit is None:
            raise ValueError("have not processes an image yet")
        
        left_curve_rad = measure_radius_of_curvature(left_x_predictions)
        right_curve_rad = measure_radius_of_curvature(right_x_predictions)
        average_curve_rad = (left_curve_rad + right_curve_rad)/2
        curvature_string = "Radius of curvature: %.2f m" % average_curve_rad

        # compute the offset from the center
        lane_center = (right_x_predictions[-1] + left_x_predictions[-1])/2
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        center_offset_pixels = abs(img_size[0]/2 - lane_center)
        center_offset_mtrs = xm_per_pix*center_offset_pixels
        offset_string = "Center offset: %.2f m" % center_offset_mtrs
        return curvature_string, offset_string

    def process(self, image):
        '''
        find the lane lines on the transforemd images
        :param image: a perspective transformed binary warped image
        :return: images with lane line highlighted
        '''
        self.img_size = image.shape
        if self.left_fit is None or self.right_fit is None:
            leftx, lefty, rightx, righty, _ = self.slidingWindowFindLanePixel(image)
        else:
            leftx, lefty, rightx, righty, _ = self.searchAroundPoly(image)

        left_fitx, right_fitx, ploty = self.fitPoly(image.shape, leftx, lefty, rightx, righty)
        
        self.curve_msg, self.offset_msg = self.calculateMetaData(image.shape, left_fitx, right_fitx)
        ### Visualization ##
        # Create an image to draw on and an image to show the selection window
        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


        ## End visualization steps ##


        return color_warp