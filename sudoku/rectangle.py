import cv2 as cv
import numpy as np
import torch
import torchvision
from torch import nn

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(True),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


net = Net().cuda()
net.load_state_dict(torch.load('feedforward.ckpt'))
transforms = torchvision.transforms.ToTensor()

# 分割
x_step, y_step = int((x_max - x_min) / 9), int((y_max - y_min) / 9)
print(x_step, y_step)
tiles = []
nums = []
for i in range(9):
    for j in range(9):
        tile = sudoku[i * y_step:(i + 1) * y_step, j * x_step:(j + 1) * x_step]
        tile = tile[8:-8, 8:-8]
        tile = cv.resize(tile, (28, 28))
        tile = cv.bitwise_not(tile)
        tiles.append(tile)
        if cv.countNonZero(tile) == 0:
            nums.append(0)
            continue
        img = transforms(tile)
        img = img.reshape(-1, 28 * 28).cuda()
        output = net(img)
        _, predicted = torch.max(output, 1)
        nums.append(predicted.cpu().numpy()[0])


tiles = np.hstack(tiles)
img = np.vstack(np.hsplit(tiles, 9))
cv.imshow("img", img)

print(np.array(nums).reshape((-1, 9)))

key = cv.waitKey()
if key == ord('q'):
    print(key)
