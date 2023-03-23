from SIFT.Pyramid.buildScaleSpace import get_img_base, get_octaves_n, get_gaussian_kernels, get_LoG, get_DoG
from SIFT.Keypoint.extrema import peakDetection
from SIFT.Keypoint.extrema import findScaleSpaceExtrema
from SIFT.Keypoint.clean import deduplicate_keypoints, postsolve_keypoints
from SIFT.Keypoint.clean import removeDuplicateKeypoints, convertKeypointsToInputImageSize
from utils.visualization import *
from utils.stat_time import *
import numpy as np
import cv2


if __name__ == "__main__":
    instance = img_process(1258)
    float_img = instance.astype('float32')
    # orig_img = cv2.imread('/Users/lixiaoyang/Desktop/CS4186/ass1/datasets_4186/query_4186/27.jpg',0)
    # img = orig_img.astype('float32')
    timer = Timer()
    r, img = get_img_base(float_img, assumed_blur=1, octaves_n=5)
    kernels = get_gaussian_kernels()
    logs = get_LoG(img, kernels=kernels, octaves_n=5)
    dogs = get_DoG(logs)

    keypoints = peakDetection(logs, dogs)
    keypoints = deduplicate_keypoints(keypoints)
    keypoints = postsolve_keypoints(keypoints, r)
    
    timer.tac()
    sing_imshow(dogs[0][0])
    point_imshow(instance, [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints])
    point_imshow(float_img, [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints])


    





    