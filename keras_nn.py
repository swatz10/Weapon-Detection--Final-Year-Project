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
filelist = glob.glob(r'D:\FINAL YEAR\code\KnivesImagesDatabase\KnivesImagesDatabase\POSITIVES_ALL\*.bmp')
x = np.array([np.array(cv2.resize(cv2.imread(fname),(224,224))) for fname in filelist])
print(x.shape)
filelist2 = glob.glob(r'D:\FINAL YEAR\code\KnivesImagesDatabase\KnivesImagesDatabase\NEGATIVES_ALL\*.bmp')
x2 = np.array([np.array(cv2.resize(cv2.imread(fname),(224,224))) for fname in filelist2])
print(x2.shape)
data=np.concatenate((x2,x),axis=0)
label=[]
with open(r"D:\FINAL YEAR\code\label_knife_whole.csv", 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    for row in reader:          #fill array by file info by for loop
        label.append(row)
    label = np.array(label)  
print(data.shape)
X_train, X_test, Y_train, Y_test = train_test_split(data,label,test_size=0.3,shuffle=True)

#X_train = X_train.reshape(X_train.shape[0], 3, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 3, 28,28)
print(X_train.shape)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
model = Sequential()
#model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
model.add(Conv2D(64, 3, strides=(1, 1),  activation='relu',input_shape=(224,224,3),data_format="channels_last"))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(64, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
print(model.output_shape)

#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(128, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(128, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
print(model.output_shape)

#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(256, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, strides=(1, 1),  activation='relu'))
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(256, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
print(model.output_shape)

#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(512, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(512, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(512, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
print(model.output_shape)

#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(512, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(512, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
#model.add(ZeroPadding2D((1,1)))
print(model.output_shape)
model.add(Conv2D(512, 3, strides=(1, 1),  activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
print(model.output_shape)

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
print("after dense 1",model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
print("after dense 2",model.output_shape)
print(model.output_shape)
model.add(Dense(2, activation='softmax'))
print("after dense 3",model.output_shape)
y_binary=to_categorical(Y_train)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.fit(X_train, y_binary, nb_epoch=25, verbose=1)
print("training done")
y_bin_test=to_categorical(Y_test)
score = model.evaluate(X_test, y_bin_test, verbose=1)
print(score)


# serialize model to JSON



"""
model_json = model.to_json()
with open(r"D:\FINAL YEAR\code\model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"D:\FINAL YEAR\code\model.h5")
print("Saved model to disk")

json_file = open(r"D:\FINAL YEAR\code\model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"D:\FINAL YEAR\code\model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_bin_test, verbose=0)
print(score)
"""