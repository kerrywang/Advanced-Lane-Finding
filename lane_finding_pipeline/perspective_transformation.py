import cv2
import constant
import matplotlib.pyplot as plt
import numpy as np
import os

from lane_finding_pipeline.piplineinterface import PipeLineInterface

class PerspectiveTransform(PipeLineInterface):
    def __init__(self, imgCoor=None, realCoor=None):
        self.imgCoor = imgCoor if imgCoor is not None else np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])
        self.realCoor = realCoor if realCoor is not None else np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])
        self.perspectiveTransformMatrix = self._findPerspectiveMatrix()
        self.revPerspectiveTransformationMatrix = self._findPerspectiveMatrix(rev=True)

    def process(self, image):
        return cv2.warpPerspective(image, self.perspectiveTransformMatrix, image.shape[1::-1])

    def unWrap(self, image):
        return cv2.warpPerspective(image, self.revPerspectiveTransformationMatrix, image.shape[1::-1])


    def _findPerspectiveMatrix(self, rev=False):
        calibrationImgPath = constant.getCameraTransformationTrainingTarget()

        if not rev:
            return cv2.getPerspectiveTransform(self.imgCoor, self.realCoor)
        return cv2.getPerspectiveTransform(self.realCoor, self.imgCoor)


if __name__ == "__main__":
    a = PerspectiveTransform()
    calibrationImgPath = os.path.join(constant.getTestImagesDir(), "test3.jpg")


    img = a.process(cv2.imread(calibrationImgPath))
    unwraped = a.unWrap(img)
    cv2.imshow('translated', img)
    cv2.imshow('unwrapped', unwraped)
    cv2.waitKey(0)
