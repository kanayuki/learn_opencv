import cv2 as cv
import numpy as np

img = cv.imread('../Image/sudoku_001.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

# canny = cv.Canny(cv.bitwise_not(binary), 50, 150)
canny = cv.Canny(gray, 50, 150)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# [[pt0,pt1,pt2,,,], [],,,], [[0后,1前,2父,3内], [],,,]
print(contours[0].shape)
rect = [cont for cont in contours if cv.contourArea(cont) > 4e4]
print(f'数量:{len(rect)}')
print(f'Shape:{rect[0].shape}')
rect = rect[0]
poly = cv.approxPolyDP(rect, 10, True)
print(f'Shape:{poly.shape}')

# 绘制轮廓矩形
copy = img.copy()
cv.polylines(copy, [poly], True, (0, 0, 255), 5)
cv.polylines(copy, [rect], True, (0, 255, 0), 2)

# cv.imshow("sudoku", canny)
# cv.imshow("binary", binary)
cv.namedWindow("Contours", cv.WINDOW_NORMAL)
cv.resizeWindow("Contours", 512, 512)
cv.imwrite("../Image/sudoku_001_rect.png", copy)
cv.imshow("Contours", copy)

# 裁剪出数独区域
# xy_min = np.min(np.squeeze(poly), 0)
# xy_max = np.max(np.squeeze(poly), 0)
# print(xy_min, xy_max)
# cv.imshow("Clip", img[xy_min[1]: xy_max[1], xy_min[0]:xy_max[0]])
x_min, y_min = np.min(np.squeeze(poly), 0)
x_max, y_max = np.max(np.squeeze(poly), 0)
sudoku = img[y_min:y_max, x_min: x_max]

cv.imwrite("../Image/sudoku_001_clip.png", sudoku)
cv.imshow("Sudoku", sudoku)

# cv.imshow("Contours", img)
key = cv.waitKey()
if key == ord('q'):
    print(key)
