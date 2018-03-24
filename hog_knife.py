#hog for knife
import glob
import cv2
import numpy as np
import csv
i=10000;
# filelist = glob.glob(r'D:\FINAL YEAR\Dataset\\*.jpg')
while (i<=19339) :
    image=cv2.imread(r"D:\FINAL YEAR\Dataset\KnivesImagesDatabase\KnivesImagesDatabase\NEGATIVES_ALL\%d.bmp" %i);
    height, width = image.shape[:2]
    if(height<100 or width<100):
        image=cv2.resize(image,(100,100))
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
    f=open(r"D:\FINAL YEAR\code\neg_gun_hog.csv","ba")
    np.savetxt(f,a,delimiter=',')
    f.close
    i=i+1;
    cv2.waitKey(0)