from lane_finding_pipeline.piplineinterface import PipeLineInterface
import cv2
import numpy as np
import constant
import os
class GradientFilter(PipeLineInterface):
    def __init__(self, filter=[]):
        self.filters = filter

    def addFilter(self, filterClass):
        assert isinstance(filterClass, GradientFilter)
        self.filters.append(filterClass)

    def _findFilter(self, orien):
        avaibaleFilter = [filter_ for filter_ in self.filters if filter_.orientation == orien]
        if not avaibaleFilter:
            return None
        return avaibaleFilter[0]

    def process(self, image):
        xyMask = np.zeros(image.shape[:2])
        magDirMask = np.zeros(image.shape[:2])

        filterX = self._findFilter('x')
        filterY = self._findFilter('y')

        filterMag = self._findFilter('mag')
        filterDir = self._findFilter('dir')

        if filterX or filterY:
            gradX = filterX.process(image) if filterX else np.ones(image.shape[:2])
            gradY = filterY.process(image) if filterY else np.ones(image.shape[:2])
            xyMask[(gradX == 1) & (gradY == 1)] = 1

        if filterMag or filterDir:
            gradMag = filterMag.process(image) if filterMag else np.ones(image.shape[:2])
            gradDir = filterMag.process(image) if filterDir else np.ones(image.shape[:2])

            magDirMask[(gradMag == 1) & (gradDir == 1)] = 1

        resultBinary = np.zeros(image.shape[:2])
        resultBinary[(xyMask == 1) | (magDirMask == 1)] = 1
        return resultBinary


class SobelFilter(GradientFilter):
    def __init__(self, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
        self.orientation = orient
        self.threshMin = thresh_min
        self.threshMax = thresh_max
        self.kenelSize = sobel_kernel

    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kenelSize)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kenelSize)
        if self.orientation == 'x':
            abs_sobel = np.absolute(sobelX)
        elif self.orientation == 'y':
            abs_sobel = np.absolute(sobelY)
        elif self.orientation == 'mag': # magnitude filter
            abs_sobel = np.sqrt(sobelX ** 2 + sobelY ** 2)
        else: # direction filter
            abs_sobel = np.arctan2(np.absolute(sobelY), np.absolute(sobelX))

        if self.orientation == 'dir':
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        else:
            scaled_sobel = abs_sobel

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= self.threshMin) & (scaled_sobel <= self.threshMax)] = 1
        return binary_output

if __name__ == "__main__":
    gf = GradientFilter()

    gf.addFilter(SobelFilter(orient='x', thresh_min=0, thresh_max=255))
    gf.addFilter(SobelFilter(orient='y', thresh_min=0, thresh_max=255))
    gf.addFilter(SobelFilter(orient='mag', sobel_kernel=9, thresh_min=0, thresh_max=255))
    gf.addFilter(SobelFilter(orient='dir', thresh_min=0, thresh_max=np.pi/2))

    binary = gf.process(cv2.imread(os.path.join(constant.getTestImagesDir(), "test1.jpg")))
    cv2.imshow('masked', binary)
    cv2.waitKey(0)