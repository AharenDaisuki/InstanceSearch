import numpy as np
import cv2
import imutils
import sys

# generate histogram
def histogram(img, mask, bins_n = (8,12,3)):
    hist = cv2.calcHist([img], [0,1,2], mask, bins_n, [0,180,0,256,0,256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist).flatten()
    else:
        hist = cv2.normalize(hist,hist).flatten()
    return hist

def feature(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    (height, width) = img.shape[:2]
    (x,y) = (int(width * 0.5), int(height * 0.5))
    segments = [(0,x,0,y),(x,width,0,y),(x,width,y,height),(0,x,y,height)]
	#center (ellipse shape)
    (x_,y_) = (int(width*0.75)//2, int(height*0.75)//2) #axes length
	#elliptical black mask
    ellipMask = np.zeros(img.shape[:2],dtype= "uint8")
    cv2.ellipse(ellipMask,(x,y),(x_,y_),0,0,360,255,-1)  # -1 :- fills entire ellipse with 255(white) color

    for (x0,x1,y0,y1) in segments:
        mask = np.zeros(img.shape[:2],dtype='uint8')
        cv2.rectangle(mask,(x0,y0),(x1,y1),255,-1)
        mask = cv2.subtract(mask, ellipMask)
        hist = histogram(img, mask)
        features.extend(hist)
    
    hist = histogram(img, ellipMask)
    features.extend(hist)
    return features

def generate_color_feature_file(img, file_path):
    sys.stdout = open(file_path, 'w')
    print(feature(img))
    sys.stdout.close()
    sys.stdout = sys.__stdout__

def similarity(hist1, hist2, eps=1e-10):
    return 0.5 * np.sum([(x-y)**2/(x+y+eps) for (x,y) in zip(hist1,hist2)])