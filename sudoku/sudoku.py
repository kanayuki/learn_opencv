import cv2 as cv
import numpy as np
import torch
import torchvision
import win32gui
from PIL import ImageGrab
from sudoku_cnn import CNN


class Sudoku:
    def __init__(self, *args, **kwargs):
        # if len(args) == 0:
        #     img = self.grab_win()
        # else:
        #     img = cv.imread(args[0])
        #
        self.origin_img = None
        self.clip_region = None
        self.sudoku_img = None
        self.sudoku_data = None
        self.cnn = None
        self.tiles = None

        # self.find_sudoku()

    def save_img(self, path):
        cv.imwrite(path, self.origin_img)

    def grab_win(self):
        handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
        x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
        self.bbox = [x1, y1, x2 - x1, y2 - y1]
        image = ImageGrab.grab(self.bbox)
        return cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)

    def find_sudoku(self, img):
        self.origin_img = img

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
        gray = cv.bitwise_not(gray)
        canny = cv.Canny(gray, 50, 150)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cnts = [cnt for cnt in contours if cv.contourArea(cnt) > 5e4]
        if not cnts:
            print('没有找到数独轮廓')
            return False
        cv.drawContours(img, cnts, 0, (0, 0, 255), 2)
        cnt = cnts[0]
        poly = cv.approxPolyDP(cnt, 10, True)
        xmin, ymin = np.min(np.squeeze(poly), 0)
        xmax, ymax = np.max(np.squeeze(poly), 0)
        clip = gray[ymin:ymax, xmin:xmax]

        self.clip_region = [xmin, ymin, xmax, ymax]
        self.sudoku_img = clip
        # self.split_recognized()
        return True

    def recognized(self, image):
        if self.cnn is None:
            self.cnn = CNN().cuda()
            self.cnn.load_state_dict(torch.load('sudoku-cnn.ckpt'))
            self.to_tensor = torchvision.transforms.ToTensor()

        size = 30
        image = cv.resize(image, (size, size))
        img = self.to_tensor(image).reshape(1, 1, size, size).cuda()

        output = self.cnn(img)
        _, predicted = output.max(1)
        n = predicted.item() + 1
        return n

    def split_recognized(self):

        sudoku_img = self.sudoku_img
        row, col = sudoku_img.shape
        # split
        tiles = []
        nums = []
        xstep, ystep = int(col / 9), int(row / 9)
        self.xstep = xstep
        self.ystep = ystep
        e = int(ystep * 0.12)

        # t = 145
        for i in range(9):
            for j in range(9):
                tile = sudoku_img[i * ystep:(i + 1) * ystep, j * xstep:(j + 1) * xstep]
                tile = tile[e:-e, e:-e]
                tiles.append(tile)
                if np.count_nonzero(tile) == 0:
                    nums.append(0)
                    continue

                n = self.recognized(cv.GaussianBlur(tile, (5, 5), 0))
                nums.append(n)

                print('{}行 {}列 : {}'.format(i + 1, j + 1, n))

        num_img = np.hstack(tiles)
        num_img = np.vstack(np.hsplit(num_img, 9))

        nums = np.array(nums).reshape(9, 9)
        print(nums)
        self.sudoku_data = nums
        self.tiles = tiles
