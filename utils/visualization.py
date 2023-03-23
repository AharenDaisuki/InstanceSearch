import cv2
from numpy import hstack
from utils.params import DATASET_PATH, GT_DIR, QUERY_DIR

def img_process(instance):
    gt_path = DATASET_PATH + GT_DIR + '/' + str(instance) + '.txt'
    img_path = DATASET_PATH + QUERY_DIR + '/' + str(instance) + '.jpg'
    img = cv2.imread(img_path, 0)
    # ground truth box
    with open(gt_path, 'r') as f:
        line = f.readline()
        gt_box = line.split(' ')
        a, b, c, d = gt_box
        a, b, c, d = int(a), int(b), int(c), int(d)
    img = img[b:b+d, a:a+c]
    # img = img.astype('float32')
    return img


def sing_imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

def multi_imshow(img_list):
    cv2.imshow('images', hstack(img_list))
    cv2.waitKey(0)

def point_imshow(img, points, size=1, color=(0,0,255),thickness=4):
    for point in points:
        cv2.circle(img, point, size, color, thickness)
    sing_imshow(img)