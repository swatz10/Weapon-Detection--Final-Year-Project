import numpy as np
import cv2
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch

import pickle
# pickle_out=open(r"D:\FINAL YEAR\code\models (finetuned 2nd)\human_model_hog_tanh400803012.pkl","rb")
# pickle_out=open(r"D:\FINAL YEAR\code\models(fresh)\human_model_hog_tanh400803012.pkl","rb")
pickle_out=open(r'human_model_hog_tanh400803012.pkl',"rb")

mlp=pickle.load(pickle_out)
pickle_out.close()
print("loaded")
# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Rebel Without a Cause -Knife.mp4')
# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Chef Uses Her Knife.avi')
# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Guns- Is That a Badge.mp4')
# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\gun-Somebody Or Nobody.mp4')

# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Patna Junction.mp4')
# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Shopping Mall.mp4')
vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Sultanganj railway station.mp4')

count = 0
success = True
flag=0
frameno=0

# perform selective search
img=cv2.imread(r'test4.jpg')

#selective search

while success:
    print("in while")
    # print (count)
    success,img = vidcap.read()
    if(not success):
        exit()
    if(count%3==0):
        print("in success")
        print("frame:")
        print(frameno)
        frameno+=1
        print(type(img))
        rows=len(img)
        cols=len(img[0])
        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        # print(regions)
        # print(regions.type)
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])

        # draw rectangles on the original image
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(img)
        for x, y, w, h in candidates:
            print(x, y, w, h)
            seg=img[y:y+h,x:x+w]
            count+=1
            # cv2.imwrite(r"D:\FINAL YEAR\code\%d.jpg" % count, seg)
            # rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            # ax.add_patch(rect)
            winSize=(64,128)#for human
            blockSize = (16,16)
            blockStride = (8,8)
            cellSize = (8,8)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
            #compute(img[, winStride[, padding[, locations]]]) -> descriptors
            winStride = (8,8)
            padding = (8,8)
            locations = ((10,20),)
            hist=[]
            hist = hog.compute(seg,winStride,padding,locations) #4356
            a=np.asarray(hist)
            a=a.T
            a=np.array(list(a), dtype=np.float)
            y_test_probabilities = mlp.predict_proba(a)
            # print(y_test_probabilities)
            res = y_test_probabilities[:,1] > 0.5
            #res=mlp.predict(a)
            k=0
            for k in range(len(res)):
                if res[k]==1:
                    if flag==0:
                        flag=1
                        # cv2.imwrite(r"C:\frames\chef knife\human-yes\cfyes_op%d.jpg" % count, seg)
                        # cv2.imwrite(r"C:\frames\knife-rebel\human-yes\kfyes_op%d.jpg" % count, seg)
                        # cv2.imwrite(r"C:\frames\gun- somebody\human-yes\cfyes_op%d.jpg" % count, seg) 
                        # cv2.imwrite(r"C:\frames\gun- is that badge\human-yes\cfyes_op%d.jpg" % count, seg) 

                        # cv2.imwrite(r"C:\frames\Mall\human-yes\cfyes_op%d.jpg" % count, seg) 
                        # cv2.imwrite(r"C:\frames\Patna\human-yes\cfyes_op%d.jpg" % count, seg) 
                        cv2.imwrite(r"C:\frames\Sultanganj\human-yes\cfyes_op%d.jpg" % count, seg) 

                        # cv2.imwrite(r"C:\fine tuned output\knife-rebel\knife hog-yes\kryes_op%d.jpg" % count, seg) 
                        # cv2.imwrite(r"C:\fine tuned output\chef knife\human-yes\cfyes_op%d.jpg" % count, seg)
                        flag=0
                        # cv2.imshow("gun",img)
                if res[k]==0:
                    
                    # cv2.imwrite(r"C:\frames\Mall\human-no\cfno_op%d.jpg" % count, seg) 
                    # cv2.imwrite(r"C:\frames\Patna\human-no\cfno_op%d.jpg" % count, seg) 
                    cv2.imwrite(r"C:\frames\Sultanganj\human-no\cfno_op%d.jpg" % count, seg) 

                    # cv2.imwrite(r"C:\frames\chef knife\human-no\cfno_op%d.jpg" % count, seg) 
                    # cv2.imwrite(r"C:\frames\knife-rebel\human-no\kfo_op%d.jpg" % count, seg) 
                    # cv2.imwrite(r"C:\frames\gun- somebody\human-no\cfno_op%d.jpg" % count, seg) 
                    # cv2.imwrite(r"C:\frames\gun- is that badge\human-no\cfno_op%d.jpg" % count, seg)

                    # cv2.imwrite(r"C:\fine tuned output\knife-rebel\knife hog-no\krno_op%d.jpg" % count, seg)
                    # cv2.imwrite(r"C:\fine tuned output\chef knife\knife hog - no\cfno_op%d.jpg" % count, seg) 
    count+=1

        # plt.show()
    print("frame done")
