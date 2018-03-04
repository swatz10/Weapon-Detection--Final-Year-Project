from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import numpy as np
import glob
import cv2
import csv
import os
import json
import pickle


X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
with open(r"D:\FINAL YEAR\code\X_train_pretrained.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        X_train.append(row)
    X_train = np.array(X_train) 
print(1)
with open(r"D:\FINAL YEAR\code\X_test_pretrained.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        X_test.append(row)
    X_test = np.array(X_test) 
print(2)
with open(r"D:\FINAL YEAR\code\Y_train_pretrained.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        Y_train.append(row)
    Y_train = np.array(Y_train)
print(3)
with open(r"D:\FINAL YEAR\code\Y_test_pretrained.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        Y_test.append(row)
    Y_test = np.array(Y_test)
print(4)
#cross validation

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring=scoring,n_jobbs=-1, verose=1)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
