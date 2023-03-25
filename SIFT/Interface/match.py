from cv2 import FlannBasedMatcher
from utils.stat_time import Timer

def match(des1, des2):
    timer = Timer(name='matching')
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test
    good = 0
    for i, j in matches:
        print(i.distance, j.distance)
        if i.distance < 0.7 * j.distance:
            good += 1
    timer.tac()        
    return good