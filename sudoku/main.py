import sys
import time

import numpy as np
import pyautogui
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
        self.bbox = [x1, y1, x2 - x1, y2 - y1]
        image = ImageGrab.grab(self.bbox)
        return image

    def find_sudoku(self):
        img = self.origin_img
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
        self.xstep = xstep
        self.ystep = ystep
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


def click_num(n):
    locs = []
    for i in range(1, 10):
        pos = pyautogui.locateCenterOnScreen(f'btn_img/btn_{i}.png')
        locs.append(pos)


if __name__ == '__main__':
    print('started')
    handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
    win32gui.ShowWindow(handle, 4)
    win32gui.SetForegroundWindow(handle)
    # 判断是否加载中
    while True:
        time.sleep(0.5)
        if pyautogui.locateCenterOnScreen('btn_img/Menu_Back.png', confidence=0.8):
            break

    print('Sudoku loaded!')
    time.sleep(0.5)

    sudoku = Sudoku()
    sudoku.save_img('screenshot.jpg')

    np.save('data/sudoku_data-6', sudoku.sudoku_data)

    solver = SudokuSolver(sudoku.sudoku_data.copy())
    solve = solver.s()
    print(solve)
    print(solver.check())

    # 自动填入

    locs = []
    for i in range(1, 10):
        pos = pyautogui.locateCenterOnScreen(f'btn_img/btn_{i}.png', confidence=0.8)
        locs.append(pos)
        # pyautogui.click(*pos)

    rx = sudoku.bbox[0] + sudoku.clip_region[0]
    ry = sudoku.bbox[1] + sudoku.clip_region[1]
    pos_blank = rx + sudoku.xstep * 10, ry + sudoku.ystep / 2
    print(pos_blank)
    time.sleep(0.1)
    pyautogui.click(*pos_blank)
    time.sleep(0.1)

    data = solve - sudoku.sudoku_data
    print('fill', data)
    for ind in range(9):
        pyautogui.click(*locs[ind])
        time.sleep(0.3)
        poss = np.argwhere(data == (ind + 1))
        for i, j in poss:
            pos = rx + (j + 0.5) * sudoku.xstep, ry + (i + 0.5) * sudoku.ystep
            pyautogui.click(*pos)
            time.sleep(0.1)

    pyautogui.click(*pos_blank)
    time.sleep(0.2)
