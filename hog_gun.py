#hog for gun
import numpy as np
import csv
import cv2
import os
folder=r'D:\FINAL YEAR\Dataset\GunDataset'
count=0
for filename in os.listdir(folder):
    image = cv2.imread(os.path.join(folder,filename))
    print(type(image))
    winSize = (88,88)
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (4,4)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist=[]
    hist = hog.compute(image,winStride,padding,locations) #4356
    a=np.asarray(hist)
    a=a.T
    f=open(r"D:\FINAL YEAR\code\pos_gun_hog.csv","ba")
    np.savetxt(f,a,delimiter=',')
    count+=1
    print(count)
    f.close
    cv2.waitKey(0)