import numpy as np
import cv2
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
pickle_out=open(r"D:\FINAL YEAR\code\knife_model.pkl","rb")
mlp=pickle.load(pickle_out)
pickle_out.close()
print("loaded")
img = imread(r'D:\FINAL YEAR\code\test.jpg')
#img=cv2.resize(img,(640,1023))
flag=0
print(type(img))
rows=len(img)
cols=len(img[0])
print(rows)
print(cols)
i=0
flag=0
count=0
for i in range(0, rows, 10):
    j=0
    for j in range(0, cols, 10):
        seg=img[i:i+100,j:j+100]
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
        hist = hog.compute(seg,winStride,padding,locations) #4356
        a=np.asarray(hist)
        a=a.T
        a=np.array(list(a), dtype=np.float)
        y_test_probabilities = mlp.predict_proba(X_test)
        res = y_test_probabilities[:,1] > 0.7
        #res=mlp.predict(a)
        k=0
        for k in range(len(res)):
            if res[k]==1:
                if flag==0:
                    flag=1
                    r=i
                    c=j
                    print(r)
                    print(c)
        count+=1
        j+=10
    i+=10
print(flag)