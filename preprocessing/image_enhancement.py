import cv2
import numpy as np

def contrast_adjustment(images, alpha=1):
    X = []
    for img in images:
        img = cv2.convertScaleAbs(img, alpha=alpha)
        pixels = img.reshape(1, -1)

        if len(X) == 0:
            X = pixels
        else:
            X = np.vstack([X, pixels])
    return X

def bilateral_filter(images, d=9, sigmaColor=75, sigmaSpace=75):
    X = []
    for img in images:
        img = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        pixels = img.reshape(1, -1)

        if len(X) == 0:
            X = pixels
        else:
            X = np.vstack([X, pixels])
    return X

def histogram_equalization(images):
    X = []
    for img in images:
        img = cv2.equalizeHist(img)
        pixels = img.reshape(1, -1)

        if len(X) == 0:
            X = pixels
        else:
            X = np.vstack([X, pixels])
    return X

def gaussian_blur(images, kernel_size=(5,5), sigmaX=0):
    X = []
    for img in images:
        img = cv2.GaussianBlur(img, kernel_size, sigmaX)
        pixels = img.reshape(1, -1)

        if len(X) == 0:
            X = pixels
        else:
            X = np.vstack([X, pixels])
    return X