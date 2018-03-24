import numpy as np
import cv2
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
pickle_out=open(r"D:\FINAL YEAR\code\models\knife_model_hog_tanh400803012.pkl","rb")
mlp=pickle.load(pickle_out)
pickle_out.close()
image=cv2.imread(r'D:\FINAL YEAR\code\test3.jpg')
image=cv2.resize(image,(100,100))
flag=0
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
a=np.array(list(a), dtype=np.float)
                # y_test_probabilities = mlp.predict_proba(a)
                # # print(y_test_probabilities)
                # res = y_test_probabilities[:,1] > 0.5
res=mlp.predict(a)
print(res[0])
if(res[0]==1.0):
    print("yes")



