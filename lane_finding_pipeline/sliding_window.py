import numpy as np
import cv2
from lane_finding_pipeline.piplineinterface import PipeLineInterface

class LaneFinding(PipeLineInterface):
    def __init__(self, nwindow=9, nmargin=100, minpix=50):
        self.nwindows = nwindow
        self.nmargin = nmargin
        self.minpix = minpix

    def findHistogram(self, image):
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return histogram, leftx_base, rightx_base


    def slidingWindowFindLanePixel(self, image):
        out_img = np.dstack((image, image, image))

        histogram, leftx_base, rightx_base = self.findHistogram(image)

        window_height = np.int(image.shape[0] // self.nwindows)
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.nmargin
            win_xleft_high = leftx_current + self.nmargin
            win_xright_low = rightx_current - self.nmargin
            win_xright_high = rightx_current + self.nmargin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def process(self, image):
        '''
        find the lane lines on the transforemd images
        :param image: a perspective transformed binary warped image
        :return: images with lane line highlighted
        '''
        leftx, lefty, rightx, righty, out_img = self.slidingWindowFindLanePixel(image)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img