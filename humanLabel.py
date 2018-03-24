import numpy as np
import csv

# f = open(r"D:\FINAL YEAR\code\label_gun_whole.csv","a",newline="")
arr=[]
# arr=np.asarray(np.repeat(0,5719)) #neg
# arr1=np.asarray(np.repeat(1,6444)) #pos

# arr=np.asarray(np.repeat(0,9340)) #neg gun
arr1=np.asarray(np.repeat(1,1397)) #pos gun


np.savetxt("D:\FINAL YEAR\code\label_gun_pos.csv", arr1, delimiter=",")


# for a in arr:
#     f = open(r"D:\FINAL YEAR\code\label_human_whole.csv","a",newline="")   
#     writer = csv.writer(f)
#     writer.writerow(a)
#     f.close