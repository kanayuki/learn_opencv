import cv2
import numpy as np

path = r"C:\Users\HuiLing\Desktop\FWnd-jwaIAEpaNG.jpg"
img = cv2.imread(path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def refresh(val):
    h_min = cv2.getTrackbarPos('Hue Max', 'Beauty Mask')
    h_max = cv2.getTrackbarPos('Hue Min', 'Beauty Mask')
    s_min = cv2.getTrackbarPos('Sat Min', 'Beauty Mask')
    s_max = cv2.getTrackbarPos('Sat Max', 'Beauty Mask')
    v_min = cv2.getTrackbarPos('Val Min', 'Beauty Mask')
    v_max = cv2.getTrackbarPos('Val Max', 'Beauty Mask')
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('Beauty Mask', mask)


cv2.namedWindow('Beauty Mask', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Hue Max', 'Beauty Mask', 0, 255, refresh)
cv2.createTrackbar('Hue Min', 'Beauty Mask', 0, 255, refresh)
cv2.createTrackbar('Sat Min', 'Beauty Mask', 0, 255, refresh)
cv2.createTrackbar('Sat Max', 'Beauty Mask', 0, 255, refresh)
cv2.createTrackbar('Val Min', 'Beauty Mask', 0, 255, refresh)
cv2.createTrackbar('Val Max', 'Beauty Mask', 0, 255, refresh)

cv2.namedWindow('Beauty', cv2.WINDOW_NORMAL)
cv2.imshow('Beauty', img)
cv2.namedWindow('Beauty HSV', cv2.WINDOW_NORMAL)
cv2.imshow('Beauty HSV', hsv)

cv2.waitKey(0)

cv2.destroyAllWindows()
