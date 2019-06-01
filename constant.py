import os

def getTestImagesDir():
    return os.path.join(os.path.dirname(__file__), "test_images")

def getCameraCalibrationFolderPath():
    return os.path.join(os.path.dirname(__file__), "camera_cal")

def getCameraTransformationTrainingTarget():
    return os.path.join(getTestImagesDir(), "straight_lines1.jpg")