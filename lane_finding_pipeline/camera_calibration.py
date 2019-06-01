import constant
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from lane_finding_pipeline.piplineinterface import PipeLineInterface

class CameraCalibration(PipeLineInterface):
    def __init__(self):
        self.cameraMatrix, self.distortionCoeff = self._calibrateCamera()


    def _calibrateCamera(self):
        '''
        calibrate the camera to find the radial distortion coeff and tangential distortion coeff for later image adjustment
        will use the chess board to perform such calibration
        :return:
        '''
        imagePoint = []  # the points found by open cv in the image
        objPoint = [] # the image in the real world 3D space
        imgShape = (None, None)
        for test_image in os.listdir(constant.getCameraCalibrationFolderPath()):
            dirPath = constant.getCameraCalibrationFolderPath()
            fullImgPath = os.path.join(dirPath, test_image)
            print (fullImgPath)
            img = cv2.imread(fullImgPath)

            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgShape = grayscale.shape[::-1]

            # some image is 9, 6 and the others are 9, 5
            for xCount, yCount in [(9, 6)]:
                ret, corners = cv2.findChessboardCorners(grayscale, (xCount, yCount), None)

                if ret == True:
                    objp = np.zeros((xCount * yCount, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:xCount, 0:yCount].T.reshape(-1, 2)
                    imagePoint.append(corners)
                    objPoint.append(objp)
                    print ("found")
                    break

        # calibrate the image
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoint, imagePoint, imgShape, None, None)
        return mtx, dist


    def process(self, image):
        dst = cv2.undistort(image, self.cameraMatrix, self.distortionCoeff, None, self.cameraMatrix)

        cv2.imshow('res', dst)
        cv2.waitKey(0)


if __name__ == "__main__":
    c = CameraCalibration()
    c.process(cv2.imread(os.path.join(constant.getCameraCalibrationFolderPath(), "calibration1.jpg")))