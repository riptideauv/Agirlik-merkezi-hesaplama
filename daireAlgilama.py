import cv2
import imutils
from collections import deque
import numpy as np

video_file = ''  # ''ab03.mp4'  # if given frames are read from file
WIDTH = 800  # width of the windows
NO_OF_POINTS=50
# ONLY_MAX = False  # if True only the max circle is drawn
GREEN_RANGE = ((29, 86, 6), (64, 255, 255))
RED_RANGE = ((139, 0, 0), (255, 160, 122))
BLUE_RANGE = ((110, 50, 50), (130, 255, 255))
ORANGE_RANGE = ((160, 100, 47), (179, 255, 255))
YELLOW_RANGE = ((10, 100, 100), (40, 255, 255))

colorLower, colorUpper = BLUE_RANGE  # select color range

if len(video_file) == 0:
    kamera = cv2.VideoCapture(0)  # default web camera=0
else:
    kamera = cv2.VideoCapture(video_file)  # read from file

pts= deque(maxlen=NO_OF_POINTS)
cv2.namedWindow('frame')
cv2.moveWindow('frame', 10, 10)  # 'frame' window position
# cv2.namedWindow('mask')
# cv2.moveWindow('mask', WIDTH + 50, 10)  # 'mask' window position
while True:
    (ok, frame) = kamera.read()

    # if filename is given but frames cannot be read then exit
    if len(video_file) > 0 and not ok:
        break

    frame = imutils.resize(frame, WIDTH)
    hsv = cv2.GaussianBlur(frame, (25,25), 0)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    mask_copy = mask.copy()

    contours = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(contours) > 0:
        cmax = max(contours, key=cv2.contourArea)
        for ctr in contours:
            (x, y), radius = cv2.minEnclosingCircle(cmax)
            mts = cv2.moments(cmax)
            center = int(mts['m10']/mts['m00']),int(mts['m01']/mts['m00'])
            if radius >= 40:  # draw circle if radius>40 px
                cv2.circle(frame, (int(x), int(y)),
                           int(radius), (255, 255, 0), 4)
            pts.appendleft(center)
            # draw tracked points in deque
            for i in range(1,len(pts)):
                if pts[i] and pts[i-1]:
                    # thickness=3
                    thickness = int(np.sqrt(NO_OF_POINTS / float(i + 1)) * 1.1)
                    cv2.line(frame,pts[i-1],pts[i],(0,255,255),thickness)


    cv2.imshow("frame", frame)
    # cv2.imshow("mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# release all objects and free memory
kamera.release()
cv2.destroyAllWindows()