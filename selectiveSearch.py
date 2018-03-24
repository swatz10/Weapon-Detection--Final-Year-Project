import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


import cv2
import numpy as np

def main():
    count=0
    # loading astronaut image
    img = cv2.imread(r'D:\FINAL YEAR\code\test4.jpg')
    # print(img.shape)
    

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
        if w / h > 1.1 or h / w > 1.1:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        seg=img[y:y+h,x:x+w]
        count+=1
        cv2.imwrite(r"D:\FINAL YEAR\code\%d.jpg" % count, seg)
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()