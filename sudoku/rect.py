import cv2 as cv
import numpy as np
from numpy import cos, sin, pi

img = cv.imread('../Image/sudoku_001.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

# canny = cv.Canny(cv.bitwise_not(binary), 50, 150)
canny = cv.Canny(gray, 50, 150)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# [[pt0,pt1,pt2,,,], [],,,], [[0后,1前,2父,3内], [],,,]
print(contours[0].shape)

# for line in lines:
#     rho, theta = line[0]
#     if abs(theta - pi / 2) < 0.01 or theta == 0:
#         p = (rho * cos(theta), rho * sin(theta))
#         pt1 = (int(p[0] - 1000 * sin(theta)), int(p[1] - 1000 * cos(theta)))
#         pt2 = (int(p[0] + 1000 * sin(theta)), int(p[1] + 1000 * cos(theta)))
#         cv.line(img, pt1, pt2, (255, 0, 255), 2)
#         print(rho, theta)
#
# cv.imshow("sudoku", canny)
# cv.imshow("binary", binary)
cv.namedWindow("Contours", cv.WINDOW_NORMAL)
cv.resizeWindow("Contours", 512, 512)


def on_change(val):
    # idx = cv.getTrackbarPos("id", "Contours")
    # print(f'=={val}=={idx}==')
    copy = img.copy()
    cv.drawContours(copy, contours, val, (0, 0, 255), 3)

    # show area
    area = cv.contourArea(contours[val])
    # print(area)
    cv.putText(copy, str(area), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    cv.imshow("Contours", copy)


cv.createTrackbar("id", "Contours", -1, len(contours) - 1, on_change)

# cv.imwrite("../Image/sudoku_001_line.png", img)
cv.imshow("Contours", img)
key = cv.waitKey()
if key == ord('q'):
    print(key)
