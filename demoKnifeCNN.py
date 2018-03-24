import numpy as np
import cv2
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


# keras imports
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, Activation

# other imports
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import pickle
# pickle_out=open(r"D:\FINAL YEAR\code\models\pretrained_inception_knife_logistic.pkl","rb")
pickle_out=open(r"D:\FINAL YEAR\code\models (not finetuned)\pretrained_inception_knife_logistic.pkl","rb")
logreg=pickle.load(pickle_out)
pickle_out.close()
print("loaded")

base_model =InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=(299,299,3)), pooling=None, classes=2)

# vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Rebel Without a Cause -Knife.mp4')
vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Chef Uses Her Knife.avi')
count = 0
success = True
flag=0
frameno=0
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
            seg=cv2.resize(seg,(299,299))
            x = image.img_to_array(seg)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = base_model.predict(x)
            # print(feature.shape)
            flat = feature.flatten() 
            print(flat.shape)
            flat=[flat]
            print(flat.shape)
            res = logreg.predict(flat)
            print(res)
            k=0
            for k in range(len(res)):
                if res[k]==1:
                    if flag==0:
                        flag=1
                        # cv2.imwrite(r"C:\frames\knife-rebel\knife hog- yes\cfyes_op%d.jpg" % count, seg) 
                        cv2.imwrite(r"C:\frames\chef knife\knife cnn- yes\cfyes_op%d.jpg" % count, seg)
                        flag=0
                        # cv2.imshow("gun",img)
                if res[k]==0:
                    # cv2.imwrite(r"C:\frames\knife-rebel\knife hog- yes\cfno_op%d.jpg" % count, seg) 
                    cv2.imwrite(r"C:\frames\chef knife\knife cnn- no\cfno_op%d.jpg" % count, seg)
    count+=1

        # plt.show()
    print("frame done")
