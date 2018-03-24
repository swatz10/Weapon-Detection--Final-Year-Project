#hog for human
import pickle
import cv2
import numpy as np
import glob
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


filelist = glob.glob(r'D:\FINAL YEAR\Dataset\Human Negative\*.jpg')
filelist2 = glob.glob(r'D:\FINAL YEAR\Dataset\PETA\*.bmp')
filelist3=glob.glob(r'D:\FINAL YEAR\Dataset\PETA\*.jpg')
# label =[]
# with open(r"D:\FINAL YEAR\code\label_human_whole.csv",'r') as f:
#     reader = csv.reader(f)     
#     for row in reader:          #fill array by file info by for loop
#         label.append(row)
#     label = np.array(label)
# print("label done")
data=[]
"""
for fname in filelist:
    # print(fname)
    image=cv2.imread(fname)
    height, width = image.shape[:2]
    if(height<1000 or width<1000):
        image=cv2.resize(image,(640,480))
    winSize=(64,128)#for human
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
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
    f=open(r"D:\FINAL YEAR\code\negative_data_human.csv","ba")
    np.savetxt(f,a,delimiter=',')
    f.close
    
    
    # f=open(r"D:\FINAL YEAR\code\negative_data_human.csv","ba")
    # np.savetxt(f,a,delimiter=',')
    # f.close
    cv2.waitKey(0)

print("neg done")
count=0
for fname in filelist2:
    image=cv2.imread(fname)
    height, width = image.shape[:2]
    if(height<1000 or width<1000):
        image=cv2.resize(image,(640,480))
    winSize=(64,128)#for human
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
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
    f=open(r"D:\FINAL YEAR\code\positive_data_human.csv","ba")
    np.savetxt(f,a,delimiter=',')
    f.close

    count+=1
    print(count)
    if(count==5000):
        break;
    
    # f=open(r"D:\FINAL YEAR\code\positive_data_human.csv","ba")
    # np.savetxt(f,a,delimiter=',')
    # f.close
    cv2.waitKey(0)
print("pos done")
"""
count=0
for fname in filelist3:
    image=cv2.imread(fname)
    height, width = image.shape[:2]
    if(height<1000 or width<1000):
        image=cv2.resize(image,(640,480))
    winSize=(64,128)#for human
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
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
    f=open(r"D:\FINAL YEAR\code\positive_data_human.csv","ba")
    np.savetxt(f,a,delimiter=',')
    f.close

    count+=1
    print(count)
    if(count==4000):
        break;
    
    # f=open(r"D:\FINAL YEAR\code\positive_data_human.csv","ba")
    # np.savetxt(f,a,delimiter=',')
    # f.close
    cv2.waitKey(0)
print("pos done")