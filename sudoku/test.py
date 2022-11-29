import os
import re
import time

import cv2 as cv
import numpy as np
import pyautogui
import torch
import win32api
import win32gui
from PIL import Image
from torchvision import transforms

##
# handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
# rect = win32gui.GetWindowRect(handle)
# print(rect)
#
# width = win32api.GetSystemMetrics(0)
# height = win32api.GetSystemMetrics(1)
# print(m)

# dir = '../Image'
# fs = os.listdir(dir)
# order = 100
# for name in fs:
#
#     m = re.match(r'\d+-num-(\d)\.png', name)
#     if m:
#         order += 1
#         num = m.group(1)
#
#         path = os.path.join(dir, name)
#         img = cv.imread(path)
#         img = cv.resize(img, (30, 30))
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#         cv.imwrite(os.path.join(dir, 'nums', f'{order}-num-{num}.png'), img)
#
# print('ok')
handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
win32gui.ShowWindow(handle, 4)
win32gui.SetForegroundWindow(handle)
time.sleep(3)
locs = []
for i in range(1, 10):
    pos = pyautogui.locateCenterOnScreen(f'btn_img/btn_{i}.png')
    locs.append(pos)
    pyautogui.click(*pos)


np.array().copy()
