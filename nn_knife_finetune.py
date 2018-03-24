import numpy as np
import cv2
from skimage.io import imread
from sklearn.neural_network import MLPClassifier
import pickle
import glob
import csv
import os

pickle_out=open(r"D:\FINAL YEAR\code\models\knife_model_hog_tanh400803012.pkl","rb")
mlp=pickle.load(pickle_out)
pickle_out.close()
print("loaded")

# filelist1 = glob.glob(r'C:\frames\Chef knife\knife hog- no\*.jpg')
# filelist2 = glob.glob(r'C:\frames\Chef knife\knife hog- yes\*.jpg')

filelist1 = glob.glob(r'C:\frames\knife-rebel\knife hog- no\*.jpg')
filelist2 = glob.glob(r'C:\frames\knife-rebel\knife hog- yes\*.jpg')
hist=[]
countneg=0
countpos=0
for file in filelist1:
    img=cv2.imread(file)
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
    a = hog.compute(img,winStride,padding,locations) #4356
    # a=a.T
    hist.append(a)
    countneg+=1
print("neg:")
print(countneg)
for file in filelist2:
    img=cv2.imread(file)
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
    a = hog.compute(img,winStride,padding,locations) #4356
    # a=a.T
    hist.append(a)
    countpos+=1
print("pos:")
print(countpos)

hist=np.array(hist)
hist=hist.reshape(countneg+countpos,4356)
print("hist shape :")
print(hist.shape)

labelneg=np.asarray(np.repeat(0,countneg)) 

labelpos=np.asarray(np.repeat(1,countpos))

np.savetxt("D:\FINAL YEAR\code\label_knife_neg_finetune.csv", labelneg, delimiter=",")
np.savetxt("D:\FINAL YEAR\code\label_knife_pos_finetune.csv", labelpos, delimiter=",")

print("loading labels")
label=[]
with open(r"D:\FINAL YEAR\code\label_knife_neg_finetune.csv", 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    for row in reader:          #fill array by file info by for loop
        label.append(row)
with open(r"D:\FINAL YEAR\code\label_knife_pos_finetune.csv", 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    for row in reader:          #fill array by file info by for loop
        label.append(row)
    label = np.array(label)  

print("now training")

mlp.fit(hist,label)

os.remove(r"D:\FINAL YEAR\code\label_knife_neg_finetune.csv")

os.remove(r"D:\FINAL YEAR\code\label_knife_pos_finetune.csv")


print("____________________________________________________________start picking_____________________________________________________________________")
pickle_out=open(r"D:\FINAL YEAR\code\models (finetuned 2nd)\knife_model_hog_tanh400803012.pkl","wb")
pickle.dump(mlp,pickle_out)
pickle_out.close()