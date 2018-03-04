import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import pickle
import os
import numpy as np
import theano
import lasagne
import csv as csv
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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

X_train=np.array(list(X_train), dtype=np.float)
Y_train=np.array(list(Y_train), dtype=np.float)
X_test=np.array(list(X_test), dtype=np.float)
Y_test=np.array(list(Y_test), dtype=np.float)
X_train = X_train.reshape((-1, 1, 28, 28))
X_test = X_test.reshape((-1, 1, 28, 28))
Y_train = Y_train.astype(np.uint8)
Y_test = Y_test.astype(np.uint8)
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 28, 28),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )
# Train the network
nn = net1.fit(X_train, y_train)