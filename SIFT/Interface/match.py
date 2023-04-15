from cv2 import FlannBasedMatcher
from utils.stat_time import Timer
from utils.params import *
from utils.fileIO import *
from utils.matchUtils import *
from functools import cmp_to_key

def match(des1, des2, flann):
    # timer = Timer(name='matching')
    # index_params = dict(algorithm=0, trees=5)
    # search_params = dict(checks=50)
    # flann = FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test
    good = 0
    for i, j in matches:
        # print(i.distance, j.distance)
        if i.distance < 0.7 * j.distance:
            good += 1
    # timer.tac()        
    return good

def match_main():
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

    for i in range(0, 20):
        scores[i].sort(key=cmp_to_key(cmp))
        log_path = DATASET_PATH + RET_DIR + '/' + str(i) + '_test.txt' 
        sys.stdout = open(log_path, 'w')
        for j in range(0, 10):
            print(scores[i][j])
        sys.stdout.close()
        sys.stdout = sys.__stdout__

def match_single(q):
    # initialize flann
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    scores = []
    des_q = lis2ndarr(read_descriptor(q))
    for d in range(0, 5020):
        if d in QUERIES:
            continue
        des_lis = read_descriptor(d)
        if len(des_lis) < 15:
            continue 
        # if len(des_lis) < 100:
        #     generate_descriptor_file(d, r=0.5)
        #     des_lis = read_descriptor(d)
        des_d = lis2ndarr(des_lis)
        score = match(des_q, des_d, flann)
        print('({},{}): {}'.format(q,d,score))
        scores.append((d, score))
    
    scores.sort(key=cmp_to_key(cmp))
    log_path = DATASET_PATH + RET_DIR + '/' + str(q) + '_test.txt' 
    sys.stdout = open(log_path, 'w')
    for i in range(0, 10):
        print(scores[i])
    sys.stdout.close()
    sys.stdout = sys.__stdout__