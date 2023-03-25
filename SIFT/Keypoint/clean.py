from numpy import array
from functools import cmp_to_key
from cv2 import KeyPoint

def cmp_keypoints(kp1, kp2):
    '''return true if kp1 < kp2'''
    if kp1.pt[0] != kp2.pt[0]: return kp1.pt[0]-kp2.pt[0]
    if kp1.pt[1] != kp2.pt[1]: return kp1.pt[1]-kp2.pt[1]
    if kp1.size != kp2.size: return kp2.size-kp1.size
    if kp1.angle != kp2.angle: return kp1.angle-kp2.angle
    if kp1.response != kp2.response: return kp2.response-kp1.response
    if kp1.octave != kp2.octave: return kp2.octave-kp1.octave
    return kp2.class_id-kp1.class_id

def deduplicate_keypoints(keypoints):
    '''sort and deduplicate keypoints'''
    # speical case
    if len(keypoints) < 2:
        return keypoints
    
    keypoints.sort(key=cmp_to_key(cmp_keypoints))
    unique = [keypoints[0]]

    for keypoint in keypoints[1:]:
        rear = unique[-1]
        if rear.pt[0] != keypoint.pt[0] or \
           rear.pt[1] != keypoint.pt[1] or \
           rear.size != keypoint.size or \
           rear.angle != keypoint.angle:
            unique.append(keypoint)
    return unique

def postsolve_keypoints(keypoints):
    '''recover original input size: say base ratio=2, recover ratio=0.5'''
    ret = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave-1) & 255) 
        ret.append(keypoint)
    return ret