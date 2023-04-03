import cv2
from numpy import hstack
from utils.params import DATASET_PATH, GT_DIR, QUERY_DIR, DATA_DIR, QUERIES, RET_DIR

# def img_process(instance):
#     gt_path = DATASET_PATH + GT_DIR + '/' + str(instance) + '.txt'
#     img_path = DATASET_PATH + QUERY_DIR + '/' + str(instance) + '.jpg'
#     img = cv2.imread(img_path, 0)
#     # ground truth box
#     with open(gt_path, 'r') as f:
#         line = f.readline()
#         gt_box = line.split(' ')
#         a, b, c, d = gt_box
#         a, b, c, d = int(a), int(b), int(c), int(d)
#     img = img[b:b+d, a:a+c]
#     # img = img.astype('float32')
#     return img

def preprocess_img(idx, ratio=0.25):
    if idx in QUERIES:
        gt_path = DATASET_PATH + GT_DIR + '/' + str(idx) + '.txt'
        query_path = DATASET_PATH + QUERY_DIR + '/' + str(idx) + '.jpg'
        img = cv2.imread(query_path,0)
        with open(gt_path, 'r') as f:
            line = f.readline()
            gt_box = line.split(' ')
            a, b, c, d = gt_box
            a, b, c, d = int(a), int(b), int(c), int(d)
            img = img[b:b+d, a:a+c]
    else:
        data_path = DATASET_PATH + DATA_DIR + '/' + str(idx) + '.jpg'
        img = cv2.imread(data_path,0)
    
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
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

def check_result(i):
    log_path = DATASET_PATH + RET_DIR + '/' + str(i) + '.txt'
    # log_path = DATASET_PATH + RET_DIR + '/' + str(i) + '_test' + '.txt'
    with open(log_path, 'r') as f:
        for line in f:
            lis = line[1:].split(',')
            title = str(lis[0])
            display(title, lis[0])

def display(title, i):
    if i in QUERIES:
        img_path = DATASET_PATH + QUERY_DIR + '/' + str(i) + '.jpg'
    else:
        img_path = DATASET_PATH + DATA_DIR + '/' + str(i) + '.jpg'           
    img = cv2.imread(img_path)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def img_process(idx, ratio=0.25, flag=0):
    if idx in QUERIES:
        gt_path = DATASET_PATH + GT_DIR + '/' + str(idx) + '.txt'
        query_path = DATASET_PATH + QUERY_DIR + '/' + str(idx) + '.jpg'
        img = cv2.imread(query_path,flag)
        with open(gt_path, 'r') as f:
            line = f.readline()
            gt_box = line.split(' ')
            a, b, c, d = gt_box
            a, b, c, d = int(a), int(b), int(c), int(d)
            img = img[b:b+d, a:a+c]
    else:
        data_path = DATASET_PATH + DATA_DIR + '/' + str(idx) + '.jpg'
        img = cv2.imread(data_path,flag)