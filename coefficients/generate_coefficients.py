import numpy as np

def NCP_3_6():
    c1 = -np.sqrt(np.sqrt(5)-2)
    c2 = -np.sqrt(2/(np.sqrt(5)-1))
    c0 = c1-c2
    return np.array([c0, c1, c2]), 0.473

def NCP_4_10():
    c1 = 0.4920434066428167763156
    c2 = -1.569846260451462851779
    c3 = -0.0340560371300231615989
    c4 = 3.007307207357765662262
    c0 = c1-c2+c3-c4
    return np.array([c0, c1, c2, c3, c4]), 0.606

