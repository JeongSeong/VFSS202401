from PIL import Image
import numpy as np
import os

def addJpg(x):
    return x+'.jpg'

def addXJpg(x):
    return x+'x.jpg'

def addYJpg(y):
    return y+'y.jpg'

def extractName(x):
    return x.split('.')[0]

def extractPID(x): # this function is for flows 
    return x.split('.')[0][:-1] # get rid of x or y to sort PID

class imageTransform():
    def __init__(self, P_dir, mode, size, resample=Image.LANCZOS):
        self.P_dir = P_dir
        self.mode = mode
        self.size = size
        self.resample = resample
    def imageOpen(self, x):
        return Image.open(os.path.join(self.P_dir, x)).convert(self.mode)
    def resizeImage(self, x):
        return x.resize(size=self.size, resample=self.resample)

def makeArray(x):
    return np.array(x)

def normBTW_zeroN_one(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))

def normBTW_absOne(img):
    return (img/np.max(img))*2 - 1