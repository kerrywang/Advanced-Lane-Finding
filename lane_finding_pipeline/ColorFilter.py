import constant
from lane_finding_pipeline.piplineinterface import PipeLineInterface
import numpy as np
import cv2
import os

class ColorFilter(PipeLineInterface):
    def __init__(self, channel, min_thresh=0, max_thresh=255):
        self.channel = channel
        self.minThresh = min_thresh
        self.maxThresh = max_thresh

    def process(self, image):
        rgb = ['R', 'G', 'B']
        hsl = ['H', 'S', 'L']

        if self.channel in rgb:
            imageBase = image[:, :, rgb.index(self.channel)]
        elif self.channel in hsl:
            imageBase = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            imageBase = imageBase[:, :, hsl.index(self.channel)]
        else:  # gray scale image
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


        # return imageBase
        # result = np.zeros_like(imageBase)
        # result[(imageBase > self.minThresh) & (imageBase <= self.maxThresh)] = 1.0
        #
        # return np.float64(result)

if __name__ == "__main__":
    image = cv2.imread(os.path.join(constant.getTestImagesDir(), "test1.jpg"))
    cf = ColorFilter('R', min_thresh=175, max_thresh=255)
    binary = cf.process(image)
    cv2.imshow('color', binary)
    cv2.waitKey(0)