            
import numpy as np
import cv2
import csv
i=10000
while (i<=13558) :
    image=cv2.imread(r"%d.bmp" %i);
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_image=cv2.resize(gray_image,(64,64))
    print(gray_image.shape)
    a=np.array(gray_image).flatten()
    f = open(r"D:\6th Sem\PR PROJECT\00_PRProj_code\full_data.csv","a",newline="")   
    writer = csv.writer(f)
    writer.writerow(a)
    f.close
    i=i+1;
    print(i)
    cv2.waitKey(0)