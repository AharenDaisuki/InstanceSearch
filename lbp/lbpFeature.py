import numpy as np

def get_array(img, x, y):
    '''return sum (1*8) given image np array'''
    ret = []
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for neighbor in neighbors:
        (i,j) = neighbor
        if img[x+i,y+j] > img[x,y]:
            ret.append(1)
        else:
            ret.append(0)
    assert(len(ret) == 8)
    return ret    

def get_sum(x):
    ret = 0
    while x:
        x &= (x-1)
        ret += 1
    return ret

def get_base(img):
    base = np.zeros(img.shape, np.uint8)
    w, h = img.shape[0], img.shape[1]
    for i in range(1, w-1):
        for j in range(1, h-1):
            arr = get_array(img, i, j)
            digit_n, ret = 0, 0
            for flag in arr:
                ret += flag<<digit_n
                digit_n += 1
            base[i,j] = ret
    return base
                

def get_lbp(img):
    uniform_rotation = np.zeros(img.shape, np.uint8)
    base = get_base(img)
    w, h = img.shape[0], img.shape[1]
    for i in range(1, w-1):
        for j in range(1, h-1):
            k = base[i,j] << 1
            if k > 255:
                k = k-255
            xor = base[i,j] ^ k
            digit_sum = get_sum(xor)
            uniform_rotation[i,j] = get_sum(base[i,j]) if digit_sum<=2 else 9
    return uniform_rotation.astype('float32')