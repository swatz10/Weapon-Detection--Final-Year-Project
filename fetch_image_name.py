from PIL import Image
import os, os.path
import numpy as np
imgs = []
path = r'D:\FINAL YEAR\code\GunDataset\*.jpg'
#valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f))
fo=open(r"D:\FINAL YEAR\code\gun_images_list.csv","ba")
np.savetxt(fo,a,delimiter=',')
fo.close