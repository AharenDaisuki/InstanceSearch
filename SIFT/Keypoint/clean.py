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

def postsolve_keypoints(keypoints, ratio):
    '''recover original input size: say base ratio=2, recover ratio=0.5'''
    ret = []
    ratio = 1.0 / ratio
    for keypoint in keypoints:
        keypoint.pt = tuple(ratio * array(keypoint.pt))
        keypoint.size *= ratio
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave-1) & 255) 
        ret.append(keypoint)
    return ret

##############################
# Duplicate keypoint removal #
##############################

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

#############################
# Keypoint scale conversion #
#############################

def convertKeypointsToInputImageSize(keypoints, ratio):
    """Convert keypoint point, size, and octave to input image size
    """
    ratio = 1.0 / ratio
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(ratio * array(keypoint.pt))
        keypoint.size *= ratio
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints