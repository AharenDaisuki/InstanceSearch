from SIFT.Interface.getKeypointsDescriptors import get_keypoints_descriptors
from SIFT.Interface.match import match, match_main
from utils.visualization import *
from utils.stat_time import *
from utils.params import *
from utils.fileIO import *
from utils.matchUtils import *
from functools import cmp_to_key
import numpy as np
import cv2

# best method: SIFT
# For more instructions, see README

def rank_list():
    # initialize flann
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)    

    scores = [[] for i in range(0, 20)] # 20 * 5000
    for i, q in enumerate(QUERIES):
        des_q = lis2ndarr(read_descriptor(q))
        for d in range(0, 5020):
            if d in QUERIES:
                continue
            des_lis = read_descriptor(d)
            if len(des_lis) < 15:
                continue
            des_d = lis2ndarr(des_lis)
            score = match(des_q, des_d, flann)
            print('({},{}): {}'.format(q,d,score))
            scores[i].append((d, score))

    log_path = DATASET_PATH + '/' + 'rankList.txt'
    sys.stdout = open(log_path, 'w')      
    
    for i in range(0, 20):
        line = 'Q'+str(i+1)+':'
        scores[i].sort(key=cmp_to_key(cmp))
        for j in scores[i]:
            line += (' ' + str(j[0]))
        print(line)

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    # match_main()
    rank_list()




    





    