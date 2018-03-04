import numpy as np
import cv2
import csv
import h5py
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras import optimizers
from keras.utils import to_categorical
from keras.utils import np_utils
import glob
import pickle

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, Activation

# other imports
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import numpy as np
import glob
import cv2
import h5py
import os
import json
import pickle




base_model =InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=(299,299,3)), pooling=None, classes=2)
# image_size = (299, 299)

# base_model =InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=(4356,1,1)), pooling=None, classes=2)

print ("[INFO] successfully loaded base model and model...")

features=[]

filelist = glob.glob(r'D:\FINAL YEAR\code\KnivesImagesDatabase\KnivesImagesDatabase\NEGATIVES_ALL\*.bmp')
filelist2 = glob.glob(r'D:\FINAL YEAR\code\KnivesImagesDatabase\KnivesImagesDatabase\POSITIVES_ALL\*.bmp')



count=1

for fname in filelist:
  img=cv2.resize(cv2.imread(fname),(299,299))
  x = image.img_to_array(img)
#   print(x.shape)
  x = np.expand_dims(x, axis=0)
#   print(x.shape)
  
  x = preprocess_input(x)
#   print(x.shape)
  
  feature = base_model.predict(x)
  # print(feature.shape)
  flat = feature.flatten()  
  # print(flat.shape)
  features.append(flat)
  # print(flat)
  print(count)
  count+=1
  
for fname in filelist2:
  img=cv2.resize(cv2.imread(fname),(299,299))
  x = image.img_to_array(img)
  # print(x.shape)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  # print(x.shape)
  
  feature = base_model.predict(x)
  # print(feature.shape)
  flat = feature.flatten()  
  # print(flat.shape)
  features.append(flat)
  print(count)
  count+=1


label=[]
with open(r"D:\FINAL YEAR\code\label_knife_whole.csv", 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    for row in reader:          
        label.append(row)
    label = np.array(label) 

X_train, X_test, Y_train, Y_test = train_test_split(features,label,test_size=0.3,shuffle=True) 

"""
with open(r"D:\FINAL YEAR\code\X_train_pretrained.csv",'w') as f:
    for line in X_train:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

with open(r"D:\FINAL YEAR\code\Y_train_pretrained.csv",'w') as f:
    for line in Y_train:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)

with open(r"D:\FINAL YEAR\code\X_test_pretrained.csv",'w') as f:
    for line in X_test:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)     

with open(r"D:\FINAL YEAR\code\Y_test_pretrained.csv",'w') as f:
    for line in Y_test:
        a=''
        for item in line:
            a+=str(item)
            a+=","
        #a=a+"0"
        a=a.rstrip(',')
        a+="\n"
        f.write(a)
"""

# pickle_out=open(r"D:\FINAL YEAR\code\X_train_pretrained.pkl","wb")
# pickle.dump(X_train,pickle_out)
# pickle_out.close()

# pickle_out=open(r"D:\FINAL YEAR\code\Y_train_pretrained.pkl","wb")
# pickle.dump(Y_train,pickle_out)
# pickle_out.close()

# pickle_out=open(r"D:\FINAL YEAR\code\X_test_pretrained.pkl","wb")
# pickle.dump(X_test,pickle_out)
# pickle_out.close()

# pickle_out=open(r"D:\FINAL YEAR\code\Y_test_pretrained.pkl","wb")
# pickle.dump(Y_test,pickle_out)
# pickle_out.close()


# print(X_train.shape)
# print(X_test.shape)

##classifer

# mlp = MLPClassifier(hidden_layer_sizes=(1000,400,80,30,12),max_iter=10000000,activation='logistic')
# X_train=np.array(list(X_train), dtype=np.float)
# Y_train=np.array(list(Y_train), dtype=np.float)
# mlp.fit(X_train,Y_train)
# print("training done!!!")
##writing fit model
# print("____________________________________________________________start picking_____________________________________________________________________")
# pickle_out=open(r"D:\FINAL YEAR\code\pretrained_inception_nn100040003012.pkl","wb")
# pickle.dump(mlp,pickle_out)
# pickle_out.close()
# X_test=np.array(list(X_test), dtype=np.float)
# Y_test=np.array(list(Y_test), dtype=np.float)
# print(Y_test.shape)

# predictions=mlp.predict(X_test)

# count=0
# for i in range(len(predictions)):
# 	if predictions[i]==Y_test[i]:
# 		count+=1
# acu=(count)/len(predictions)
# print("accuracy:")
# print(str(acu)) #accuracy


# pickle_out=open(r"D:\FINAL YEAR\code\pretrained_inception_nn100040003012.pkl","rb")
# mlp=pickle.load(pickle_out)
# pickle_out.close()
# predictions=mlp.predict(X_test)

# count=0
# for i in range(len(predictions)):
# 	if predictions[i]==Y_test[i]:
# 		count+=1
# acu=(count)/len(predictions)
# print("accuracy ater loading:")
# print(str(acu)) #accuracy


#logistic regression


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


pickle_out=open(r"D:\FINAL YEAR\code\pretrained_inception_logistic.pkl","wb")
pickle.dump(logreg,pickle_out)
pickle_out.close()


Y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))



#cross validation

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
