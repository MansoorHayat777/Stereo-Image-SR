import numpy as np
import cv2
import glob

path1 = '/'
path2 = '/'
pathout = '/'

def get_weight(path1, path2, pic_nums):
    mse1 = 0
    mse2 = 0
    path1s = sorted(glob.glob(path1 + '*.png'))
    path2s = sorted(glob.glob(path2 + '*.png'))
    for i in range(pic_nums):
        file1 = path1s[i]
        file2 = path2s[i]
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mean = (img1+img2)/2
        mse1 += np.mean((img1 - mean)**2)
        mse2 += np.mean((img2 - mean) ** 2)
    MSE = mse1 + mse2
    return mse2/MSE, mse1/MSE

def fuse(path1, path2, pic_nums):
    w1, w2 = get_weight(path1, path2, pic_nums)
    path1s = sorted(glob.glob(path1 + '*.png'))
    path2s = sorted(glob.glob(path2 + '*.png'))
    for i in range(pic_nums):
        file1 = path1s[i]
        file2 = path2s[i]
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        out = w1*img1 + w2*img2
        filename = file1.split('/')[-1]
        cv2.imwrite(pathout + filename, out)
        # break

fuse(path1, path2, 200)
