"""
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import csv as csv

data=[]
label=[]
with open(r"D:\FINAL YEAR\code\full_data_knife.csv", 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    for row in reader:          #fill array by file info by for loop
        data.append(row)
    data = np.array(data)  
with open(r"D:\FINAL YEAR\code\label_knife_whole.csv", 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    for row in reader:          #fill array by file info by for loop
        label.append(row)
    label = np.array(label)  
print(len(data))
print(len(label))
#X=data
#Y=label
X_train, X_test, Y_train, Y_test = train_test_split(data,label,test_size=0.3,shuffle=True)
print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))
###writing into dataset into file
with open(r"D:\FINAL YEAR\code\X_train_full100.csv",'w') as f:
    for line in X_train:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

with open(r"D:\FINAL YEAR\code\Y_train_full100.csv",'w') as f:
    for line in Y_train:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

with open(r"D:\FINAL YEAR\code\X_test_full100.csv",'w') as f:
    for line in X_test:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)     

with open(r"D:\FINAL YEAR\code\Y_test_full100.csv",'w') as f:
    for line in Y_test:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
with open(r"D:\FINAL YEAR\code\X_train.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        X_train.append(row)
    X_train = np.array(X_train) 
print(1)
with open(r"D:\FINAL YEAR\code\X_test.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        X_test.append(row)
    X_test = np.array(X_test) 
print(2)
with open(r"D:\FINAL YEAR\code\Y_train.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        Y_train.append(row)
    Y_train = np.array(Y_train)
print(3)
with open(r"D:\FINAL YEAR\code\Y_test.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        Y_test.append(row)
    Y_test = np.array(Y_test)
print(4)
#print(X_test)
#print(Y_test)

##classifer

mlp = MLPClassifier(hidden_layer_sizes=(400,80,30,12),max_iter=10000000,activation='tanh')
X_train=np.array(list(X_train), dtype=np.float)
Y_train=np.array(list(Y_train), dtype=np.float)
mlp.fit(X_train,Y_train)

#writing fit model
print("____________________________________________________________start picking_____________________________________________________________________")
pickle_out=open(r"D:\FINAL YEAR\code\knife_model_hog_tanh400803012.pkl","wb")
pickle.dump(mlp,pickle_out)
pickle_out.close()
X_test=np.array(list(X_test), dtype=np.float)
Y_test=np.array(list(Y_test), dtype=np.float)
print(Y_test.shape)
pickle_out=open(r"D:\FINAL YEAR\code\knife_model_hog_tanh400803012.pkl","rb")
mlp=pickle.load(pickle_out)
pickle_out.close()
predictions=mlp.predict(X_test)

count=0
for i in range(len(predictions)):
	if predictions[i]==Y_test[i]:
		count+=1
acu=(count)/len(predictions)
print(str(acu)) #accuracy
"""
import numpy as np
import cv2
from skimage.io import imread
from sklearn.neural_network import MLPClassifier
import pickle
import glob
import csv
import os

##classifer

mlp = MLPClassifier(hidden_layer_sizes=(400,80,30,12),max_iter=10000000,activation='tanh')

filelist1= glob.glob(r'D:\FINAL YEAR\Dataset\KnivesImagesDatabase\KnivesImagesDatabase\NEGATIVES_ALL\*.bmp')
filelist2= glob.glob(r'C:\frames\Chef knife\knife hog- no\*.jpg')
filelist3 = glob.glob(r'C:\frames\knife-rebel\knife hog- no\*.jpg')

filelist4= glob.glob(r'D:\FINAL YEAR\Dataset\KnivesImagesDatabase\KnivesImagesDatabase\POSITIVES_ALL\*.bmp')
filelist5 = glob.glob(r'C:\frames\Chef knife\knife hog- yes\*.jpg')
filelist6 = glob.glob(r'C:\frames\knife-rebel\knife hog- yes\*.jpg')
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
    countneg+=1
for file in filelist3:
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
for file in filelist4:
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
for file in filelist5:
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
for file in filelist6:
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
pickle_out=open(r"D:\FINAL YEAR\code\models(fresh)\knife_model_hog_tanh400803012.pkl","wb")
pickle.dump(mlp,pickle_out)
pickle_out.close()