import csv
import numpy as np

class Data:
    def __init__(self):
        with open(r"C:\Users\Sumeet\Desktop\project\Final year\code\hog\neg_knife_test.csv","r") as f:
            reader = csv.reader(f)

            data2 = np.array(list(reader))
            data2=data2.astype(np.float)

            with open(r"C:\Users\Sumeet\Desktop\project\Final year\code\hog\neg_knife_test(m).csv","r") as f:
                reader = csv.reader(f)

                data1 = np.array(list(reader))
                data1=data1.astype(np.float)

            self.data = np.concatenate((data2,data1))

            
