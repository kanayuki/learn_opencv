import cv2
import numpy as np

path = r"C:\Users\HuiLing\Desktop\FWnd-jwaIAEpaNG.jpg"
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (7, 7), 5)
cv2.GaussianBlur(img,(7,7),1)
canny = cv2.Canny(blur, 10, 10)

contours, hierarchy, = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
for cont in contours:
    area = cv2.contourArea(cont)
    if area > 500:
        print(area)
        cv2.drawContours(img, cont, -1, (0, 0, 255), 3)

cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
cv2.imshow('Contours', img)
cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

cv2.destroyAllWindows()
