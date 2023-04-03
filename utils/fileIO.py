from utils.params import *
from utils.visualization import *
from utils.fileIO import *
from SIFT.Interface.getKeypointsDescriptors import get_keypoints_descriptors
import sys

def read_descriptor(idx):
    file_path = DATASET_PATH + DES_DIR + '/' + str(idx) + '.txt'
    ret = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.split('.')
            for token in tokens:
                if token == ']\n' or token == '\n':
                    continue
                else:
                    token = token.replace(' ', '')
                    if token[0] == '[':
                        ret.append([int(token[1:])])
                    else:
                        ret[-1].append(int(token))
    return ret

def read_lbp(idx):
    file_path = DATASET_PATH + LBP_DIR + '/' + str(idx) + '.txt'
    ret = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.split('.')
            for token in tokens:
                if token == ']\n' or token == '\n':
                    continue
                else:
                    token = token.replace(' ', '')
                    if token[0] == '[':
                        ret.append([int(token[1:])])
                    else:
                        ret[-1].append(int(token))
    return ret

def read_color_feature(idx):
    file_path = DATASET_PATH + COL_DIR + '/' + str(idx) + '.txt'
    ret = [] 
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] == '[':
                line = line[1:]
            if line[-2] == ']' and line[-1] == '\n':
                line = line[:-2]
            tokens = line.split(',')
            for t in tokens:
                ret.append(float(t))
    return ret

def generate_descriptor_file(i, r=0.25):
    print("generate descriptor {} with resize ratio {}...".format(i,r))
    log_path = DATASET_PATH + DES_DIR + '/' + str(i) + '.txt' 
    sys.stdout = open(log_path, 'w')
    img = img_process(i, ratio=r)
    # kp, des = pysift.computeKeypointsAndDescriptors(img) 
    kp, des = get_keypoints_descriptors(img)
    sys.stdout.close()
    sys.stdout = sys.__stdout__

def generate_lbp_file(i):
    print("generate LBP feature {}...".format(i))
    log_path = DATASET_PATH + LBP_DIR + '/' + str(i) + '.txt'
    sys.stdout = open(log_path, 'w')
    ndarr = lis2ndarr(img_process(i))
    for feature in get_lbp(img=ndarr):
        print(feature)
    sys.stdout.close()
    sys.stdout = sys.__stdout__