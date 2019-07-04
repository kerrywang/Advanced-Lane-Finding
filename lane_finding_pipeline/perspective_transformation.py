import cv2
import constant
import matplotlib.pyplot as plt
import numpy as np
import os

from lane_finding_pipeline.piplineinterface import PipeLineInterface

class PerspectiveTransform(PipeLineInterface):
    def __init__(self):
        self.perspectiveTransformMatrix = self._findPerspectiveMatrix()
        self.revPerspectiveTransformationMatrix = self._findPerspectiveMatrix(rev=True)

    def process(self, image):
        return cv2.warpPerspective(image, self.perspectiveTransformMatrix, image.shape[1::-1])

    def unWrap(self, image):
        return cv2.warpPerspective(image, self.revPerspectiveTransformationMatrix, image.shape[1::-1])


    def _findPerspectiveMatrix(self, rev=False):
        calibrationImgPath = constant.getCameraTransformationTrainingTarget()
        imgCoor = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])
        realCoor = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])

        if not rev:
            return cv2.getPerspectiveTransform(imgCoor, realCoor)
        return cv2.getPerspectiveTransform(realCoor, imgCoor)


if __name__ == "__main__":
    a = PerspectiveTransform()
    calibrationImgPath = os.path.join(constant.getTestImagesDir(), "test3.jpg")


    img = a.process(cv2.imread(calibrationImgPath))
    unwraped = a.unWrap(img)
    cv2.imshow('translated', img)
    cv2.imshow('unwrapped', unwraped)
    cv2.waitKey(0)
