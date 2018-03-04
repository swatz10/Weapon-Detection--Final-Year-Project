from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import csv as csv
X_test=[]
Y_test=[]
with open(r"D:\FINAL YEAR\code\X_test.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        X_test.append(row)
    X_test = np.array(X_test) 
with open(r"D:\FINAL YEAR\code\Y_test.csv",'r') as f:
    reader = csv.reader(f)     
    for row in reader:          #fill array by file info by for loop
        Y_test.append(row)
    Y_test = np.array(Y_test)
X_test=np.array(list(X_test), dtype=np.float)
Y_test=np.array(list(Y_test), dtype=np.float)
print(Y_test.shape)
pickle_out=open(r"D:\FINAL YEAR\code\knife_model_hog_tanh400803012.pkl","rb")
mlp=pickle.load(pickle_out)
pickle_out.close()
#predictions=mlp.predict(X_test)
#y_test_probabilities = mlp.predict_proba(X_test)
#predictions = y_test_probabilities[:,1] > 0.7
predictions=mlp.predict(X_test)

count=0
for i in range(len(predictions)):
	if predictions[i]==Y_test[i]:
		count+=1
acu=(count)/len(predictions)
print(str(acu)) #accuracy