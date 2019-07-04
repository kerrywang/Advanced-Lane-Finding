from lane_finding_pipeline import *
import cv2
import constant
import os
import numpy as np

class ProcessPipe(object):
    def __init__(self):
        self.filters = []

    def register(self, filter):
        self.filters.append(filter)

    def process(self, image):
        result = image
        for filter in self.filters:
            result = filter.process(result)
        return result

if __name__ == "__main__":
    pipe = ProcessPipe()

    imagePath = os.path.join(constant.getTestImagesDir(), "test2.jpg")
    img = cv2.imread(imagePath)

    # camera clibration
    cmCali = CameraCalibration()
    pipe.register(cmCali)

    # color filter
    colorFilter = ColorFilter('Gray')
    pipe.register(colorFilter)

    # gradient filter
    gradFilter = GradientFilter()

    gradFilter.addFilter(SobelFilter(orient='x', thresh_min=10, thresh_max=200))
    # gradFilter.addFilter(SobelFilter(orient='y', thresh_min=20, thresh_max=100))
    # gradFilter.addFilter(SobelFilter(orient='mag', sobel_kernel=3, thresh_min=20, thresh_max=100))
    gradFilter.addFilter(SobelFilter(orient='dir', sobel_kernel=3, thresh_min=np.pi/6, thresh_max=np.pi/2))
    pipe.register(gradFilter)

    cv2.imshow('test', pipe.process(img))


    # # perspective transform
    perspective = PerspectiveTransform()
    pipe.register(perspective)

    # final process
    res = pipe.process(img)

    cv2.imshow('result', res)
    cv2.waitKey(0)


# # perspective transform
# perspective = PerspectiveTransform()
