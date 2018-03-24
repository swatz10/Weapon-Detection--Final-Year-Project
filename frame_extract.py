import cv2
vidcap = cv2.VideoCapture(r'C:\Users\Swathi\Downloads\car - gun fight.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  # image=cv2.resize(image,(200,200))
  print('Read a new frame: ', success)
  if(count%3==0):
    cv2.imwrite(r"C:\Users\Swathi\Downloads\trimmed\gunFight%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
print(count)