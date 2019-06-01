import cv2
import constant
import matplotlib.pyplot as plt
import numpy as np
import os

from lane_finding_pipeline.piplineinterface import PipeLineInterface

class PerspectiveTransform(PipeLineInterface):
    def __init__(self):
        self.perspectiveTransformMatrix = self._findPerspectiveMatrix()

    def process(self, image):
        print(image.shape[1::-1])
        cv2.imshow('original', image)
        return cv2.warpPerspective(image, self.perspectiveTransformMatrix, image.shape[1::-1])

    def _findPerspectiveMatrix(self):
        calibrationImgPath = constant.getCameraTransformationTrainingTarget()
        imgCoor = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])
        realCoor = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])

        return cv2.getPerspectiveTransform(imgCoor, realCoor)


if __name__ == "__main__":
    a = PerspectiveTransform()
    calibrationImgPath = os.path.join(constant.getTestImagesDir(), "test3.jpg")


    img = a.process(cv2.imread(calibrationImgPath))
    cv2.imshow('translated', img)
    cv2.waitKey(0)
