from SIFT.Interface.getKeypointsDescriptors import get_keypoints_descriptors
from SIFT.Interface.match import match
from utils.visualization import *
from utils.stat_time import *
from utils.params import DATASET_PATH, DATA_DIR
import numpy as np
import cv2


if __name__ == "__main__":
    query = preprocess_img(1709)
    data = preprocess_img(1234)
    # kp_q, des_q = get_keypoints_descriptors(instance)
    kp_q, des_q = get_keypoints_descriptors(query)
    for i in range(0, 1):
        kp_i, des_i = get_keypoints_descriptors(data)
        match_n = match(des_q, des_i)
        print('[{}<=>{}]: {}'.format(1709, 1234, match_n))




    





    