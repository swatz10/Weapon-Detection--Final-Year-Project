from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import csv as csv
"""
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
"""
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
