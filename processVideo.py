import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

vidcap = cv2.VideoCapture(r'D:\FINAL YEAR\code\demo videos\Chef Uses Her Knife.mp4')

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
x=1068
y=529
w=202
h=188

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('Chef Uses Her Knife.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
success= True
count=0
while success:
    success,img = vidcap.read()
    if(not success):
        exit()
    count+=1
    img[y:y+h,x:x+w]=0
    out.write(img)

# img=cv2.imread(r'frame1.jpg')
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# ax.imshow(img)
# rect = mpatches.Rectangle((1068, 529), 202,188, fill=False, edgecolor='red', linewidth=1)
# ax.add_patch(rect)
# plt.show()