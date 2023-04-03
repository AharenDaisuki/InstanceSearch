import numpy as np

def cmp(tup1, tup2):
    return tup2[1]-tup1[1]

def lis2ndarr(lis):
    return np.array(lis, dtype=np.float32)