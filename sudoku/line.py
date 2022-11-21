import cv2 as cv
import numpy as np
from numpy import cos, sin, pi

img = cv.imread('../Image/sudoku_001.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

canny = cv.Canny(cv.bitwise_not(binary), 50, 150)

lines = cv.HoughLines(canny, 0.1, np.pi / 180, 150)
for line in lines:
    rho, theta = line[0]
    if abs(theta - pi / 2) < 0.01 or theta == 0:
        p = (rho * cos(theta), rho * sin(theta))
        pt1 = (int(p[0] - 1000 * sin(theta)), int(p[1] - 1000 * cos(theta)))
        pt2 = (int(p[0] + 1000 * sin(theta)), int(p[1] + 1000 * cos(theta)))
        cv.line(img, pt1, pt2, (255, 0, 255), 2)
        print(rho, theta)

cv.imshow("sudoku", canny)
cv.imshow("binary", binary)
cv.imshow("line", img)
cv.imwrite("../Image/sudoku_001_line.png", img)
key = cv.waitKey()
if key == ord('q'):
    print(key)
