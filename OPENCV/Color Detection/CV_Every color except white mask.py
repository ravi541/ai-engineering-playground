import cv2 # package of AI
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    low = np.array([0,42,0])
    high=np.array([179,255,255])
    
    red_mask=cv2.inRange(hsv_frame,low,high)
    red = cv2.bitwise_and(frame,frame,mask = red_mask)
    
    
    
    cv2.imshow("Frame",frame)
    cv2.imshow('Red',red)
    key = cv2.waitKey(1)
    if key ==27: #27 is esc button in keyboard
        break
    
    
    