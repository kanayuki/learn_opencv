import sys

import numpy as np
import torchvision
import win32api
import win32gui
from PIL import Image, ImageGrab
from PyQt5.QtWidgets import QApplication
import cv2 as cv
import torch
from CNN import CNN
from sudoku_solver import SudokuSolver


class Sudoku:
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            img = self.grab_win()
            img = np.asarray(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        else:
            img = cv.imread(args[0])

        self.origin_img = img
        self.find_sudoku()

    def save_img(self, path):
        cv.imwrite(path, self.origin_img)

    def grab_win(self):
        handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
        x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        image = ImageGrab.grab(bbox)
        return image

    def find_sudoku(self):
        img = self.origin_img
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
        gray = cv.bitwise_not(gray)
        canny = cv.Canny(gray, 50, 150)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cnts = [cnt for cnt in contours if cv.contourArea(cnt) > 1e4]
        if not cnts:
            print('没有找到数独轮廓')
            return False
        cv.drawContours(img, cnts, 0, (0, 0, 255), 2)
        cnt = cnts[0]
        poly = cv.approxPolyDP(cnt, 10, True)
        xmin, ymin = np.min(np.squeeze(poly), 0)
        xmax, ymax = np.max(np.squeeze(poly), 0)
        clip = gray[ymin:ymax, xmin:xmax]

        self.sudoku_img = clip
        self.split_recognized()
        return True

    def split_recognized(self):

        cnn = torch.load('sudoku-cnn.ckpt').cuda()
        to_tensor = torchvision.transforms.ToTensor()

        clip = self.sudoku_img
        row, col = clip.shape
        # split
        size = 30
        tiles = []
        nums = []
        xstep, ystep = int(col / 9), int(row / 9)
        e = int(ystep * 0.12)

        # t = 145
        for i in range(9):
            for j in range(9):
                tile = clip[i * ystep:(i + 1) * ystep, j * xstep:(j + 1) * xstep]
                tile = tile[e:-e, e:-e]
                tiles.append(tile)
                if np.count_nonzero(tile) == 0:
                    nums.append(0)
                    continue

                # cnts, _ = cv.findContours(tile, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # x, y, w, h = cv.boundingRect(cnts[0])
                # min = np.min([w, h])
                # x, y = int(x + w / 2 - th / 2), int(y + h / 2 - tw / 2)
                # img = tile[y:y + th, x:x + tw]
                tile = cv.resize(tile, (size, size))

                img = to_tensor(tile).reshape(1, 1, size, size).cuda()
                output = cnn(img)
                _, predicted = output.max(1)
                n = predicted.item() + 1
                nums.append(n)

                # t += 1
                # cv.imwrite('../Image/nums/' + f'{t}-num-{n}.png', tile)

                print('{}行{}列 : {}'.format(i + 1, j + 1, n))

        num_img = np.hstack(tiles)
        num_img = np.vstack(np.hsplit(num_img, 9))

        nums = np.array(nums).reshape(9, 9)
        print(nums)
        self.sudoku_data = nums

        # cv.imshow('img', self.origin_img)
        # # cv.imshow('canny', canny)
        # cv.imshow('clip', clip)
        # cv.imshow('num', num_img)
        # cv.waitKey()


if __name__ == '__main__':
    sudoku = Sudoku()
    sudoku.save_img('screenshot.jpg')

    np.save('data/sudoku_data-6', sudoku.sudoku_data)

    solver = SudokuSolver(sudoku.sudoku_data)
    solve = solver.s()
    print(solve)
    print(solver.check())
