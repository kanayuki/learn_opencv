import cv2 as cv
import numpy as np
import torch
import torchvision
from CNN import CNN

img = cv.imread('../Image/sudoku_002.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 50, 150)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

curves = [cont for cont in contours if cv.contourArea(cont) > 4e4]
poly = cv.approxPolyDP(curves[0], 10, True)

# 裁剪出数独区域
x_min, y_min = np.min(np.squeeze(poly), 0)
x_max, y_max = np.max(np.squeeze(poly), 0)
sudoku = gray[y_min:y_max, x_min: x_max]

to_tensor = torchvision.transforms.ToTensor()

cnn = torch.load('sudoku-cnn.ckpt')

# 分割, 识别
tw, th = 38, 45
x_step, y_step = int((x_max - x_min) / 9), int((y_max - y_min) / 9)
print(x_step, y_step)
tiles = []
nums = []
for i in range(9):
    for j in range(9):
        tile = sudoku[i * y_step:(i + 1) * y_step, j * x_step:(j + 1) * x_step]
        tile = tile[8:-8, 8:-8]
        tile = cv.bitwise_not(tile)
        tiles.append(tile)
        # 识别数字
        if cv.countNonZero(tile) == 0:
            nums.append(0)
            continue

        cnts, _ = cv.findContours(tile, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv.boundingRect(cnts[0])
            # 裁剪出相同大小
            x = int(x + w / 2 - tw / 2)
            y = int(y + h / 2 - th / 2)
            tile = tile[y:y + th, x:x + tw]

            img = to_tensor(tile).reshape(1, tw * th).cuda()
            output = cnn(img)
            _, predicted = output.max(1)
            print(predicted.item() + 1)
            nums.append(predicted.item() + 1)
        else:
            raise Exception('没有发现轮廓！')

sudoku_data = np.array(nums).reshape((-1, 9))
print(sudoku_data)
np.savetxt('data/sudoku_data.txt', sudoku_data)

tiles = np.hstack(tiles)
img = np.vstack(np.hsplit(tiles, 9))
cv.imshow("img", img)

key = cv.waitKey()
if key == ord('q'):
    print(key)
