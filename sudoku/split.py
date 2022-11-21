import cv2 as cv
import numpy as np

img = cv.imread('../Image/sudoku_001.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 50, 150)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

curves = [cont for cont in contours if cv.contourArea(cont) > 4e4]
poly = cv.approxPolyDP(curves[0], 10, True)

# 裁剪出数独区域
x_min, y_min = np.min(np.squeeze(poly), 0)
x_max, y_max = np.max(np.squeeze(poly), 0)
sudoku = gray[y_min:y_max, x_min: x_max]
# sudoku = sudoku[8:-8, 8:-8]

x_step, y_step = int((x_max - x_min) / 9), int((y_max - y_min) / 9)
print(x_step, y_step)
tiles = []
for i in range(9):
    for j in range(9):
        tile = sudoku[i * y_step:(i + 1) * y_step, j * x_step:(j + 1) * x_step]
        tiles.append(tile[8:-8, 8:-8])

# print([tile.shape for tile in tiles])
print(np.array(tiles).shape)
# print(np.hstack(np.array(tiles)).shape)
# print(np.vstack(np.array(tiles)).shape)
print(np.hstack(tiles).shape)
# print(np.vstack(tiles).shape)
tiles = np.hstack(tiles)
tiles = np.vstack(np.hsplit(tiles, 9))
print(tiles.shape)

cv.imwrite("../Image/sudoku_001_tiles.png",tiles)
cv.imshow("tiles", tiles)

key = cv.waitKey()
if key == ord('q'):
    print(key)
