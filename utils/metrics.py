import numpy as np
from PIL import Image
import cv2


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0), axis=(-2, -1)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1), axis=(-2, -1)))
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1), axis=(-2, -1)))
    TN = np.float64(np.sum((prediction == 0) & (groundtruth == 0), axis=(-2, -1)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    # if accuracy is not a list
    if isinstance(accuracy, np.float64):
        accuracy = [accuracy]
    return accuracy

def iou(prediction, groundtruth):
    """Getting the IOU of the model"""
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    IOU = np.divide(TP, FP + TP + FN)
    if isinstance(IOU, np.float64):
        IOU = [IOU]
    return IOU

def show_image(image):
    cv2.imshow('image', image * 255)
    cv2.waitKey(0)


def rgb2bw(rgb_image_path, show=False):
    rgb_image = cv2.imread(rgb_image_path)

    grayimg = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    ret, bw = cv2.threshold(grayimg, 150, 1, cv2.THRESH_BINARY)

    if show:
        show_image(bw)

    return bw


if __name__ == '__main__':
    predict_image_path = r'../data/validation/labels/000040.jpg'
    ground_image_path = r'../data/validation/labels/000044.jpg'

    predict_image = np.array(rgb2bw(predict_image_path, show=False))
    ground_image = np.array(rgb2bw(ground_image_path, show=False))

    print(sum(predict_image == 1))

    accuracy = accuracy_score(predict_image, ground_image)
    IOU = iou(predict_image, ground_image)
    print(accuracy)
    print(IOU)
