#hog for human
import pickle
import cv2
import numpy as np
import glob
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


filelist = glob.glob(r'D:\FINAL YEAR\code\Human Negative\*.jpg')
filelist2 = glob.glob(r'D:\FINAL YEAR\code\Human Positive\*.jpg')

# label =[]
# with open(r"D:\FINAL YEAR\code\label_human_whole.csv",'r') as f:
#     reader = csv.reader(f)     
#     for row in reader:          #fill array by file info by for loop
#         label.append(row)
#     label = np.array(label)
# print("label done")
data=[]
count=0
for fname in filelist:
    # print(fname)
    image=cv2.imread(fname)
    height, width = image.shape[:2]
    if(height<1000 or width<1000):
        image=cv2.resize(image,(1000,1000))
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
    count+=1
    print(count)
    
    # f=open(r"D:\FINAL YEAR\code\negative_data_human.csv","ba")
    # np.savetxt(f,a,delimiter=',')
    # f.close
    cv2.waitKey(0)

print("neg done")
"""
for fname in filelist2:
    print(fname)
    image=cv2.imread(fname)
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
    
    # f=open(r"D:\FINAL YEAR\code\positive_data_human.csv","ba")
    # np.savetxt(f,a,delimiter=',')
    # f.close
    cv2.waitKey(0)
print("pos done")

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
with open(r"D:\FINAL YEAR\code\X_train_hogHuman.csv",'w') as f:
    for line in X_train:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

with open(r"D:\FINAL YEAR\code\Y_train_hogHuman.csv",'w') as f:
    for line in Y_train:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

with open(r"D:\FINAL YEAR\code\X_test_hogHuman.csv",'w') as f:
    for line in X_test:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)     

with open(r"D:\FINAL YEAR\code\Y_test_hogHuman.csv",'w') as f:
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

##writing fit model
print("____________________________________________________________start picking_____________________________________________________________________")
pickle_out=open(r"D:\FINAL YEAR\code\human_model_hog_tanh400803012.pkl","wb")
pickle.dump(mlp,pickle_out)
pickle_out.close()
X_test=np.array(list(X_test), dtype=np.float)
Y_test=np.array(list(Y_test), dtype=np.float)
print(Y_test.shape)
pickle_out=open(r"D:\FINAL YEAR\code\human_model_hog_tanh400803012.pkl","rb")
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

