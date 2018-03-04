import numpy as np
import csv

f = open(r"D:\FINAL YEAR\code\label_human_whole.csv","a",newline="")
arr=[]
# arr=np.asarray(np.repeat(0,5719)) #neg
arr1=np.asarray(np.repeat(1,19667)) #pos


np.savetxt("D:\FINAL YEAR\code\label_human_pos.csv", arr1, delimiter=",")


# for a in arr:
#     f = open(r"D:\FINAL YEAR\code\label_human_whole.csv","a",newline="")   
#     writer = csv.writer(f)
#     writer.writerow(a)
#     f.close