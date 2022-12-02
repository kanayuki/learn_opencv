import io
import time

import PySimpleGUI as sg
import numpy as np
import pyautogui
import win32gui
from PIL import ImageGrab

from sudoku import Sudoku
from main import play
from sudoku_solver import SudokuSolver
import cv2 as cv

sudoku = Sudoku()

# handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
# win32gui.ShowWindow(handle, 4)
# win32gui.SetForegroundWindow(handle)
# x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
# w, h = x2 - x1, y2 - y1
# bbox = (x1, y1, w, h)
# image = ImageGrab.grab(bbox).resize((int(w / 3), int(h / 3)))
# bytes_io = io.BytesIO()
# image.save(bytes_io, format='PNG')
# img1 = bytes_io.getvalue()
img1 = None

# image = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR+cv.COLOR_BGR2GRAY)
# img2 = cv.imencode('.png', image)[1].tobytes()
img2 = None


def img2bytes(img: np.ndarray, ext='.png'):
    ret, arr = cv.imencode(ext, img)
    if ret:
        return arr.tobytes()


sg.theme('DarkAmber')  # Add a touch of color
# All the stuff inside your window.

tab1_layout = [[sg.T('This is inside tab 1')],
               [sg.Image(img1, key='origin_img')]]

tab2_layout = [[sg.T('This is inside tab 2')],
               [sg.Image(img2, key='sudoku_img')]]

layout = [[sg.Text('Some text on Row 1')],

          # [sg.TabGroup([[sg.Tab('Tab 1', tab1_layout), sg.Tab('Tab 2', tab2_layout)]])],
          [sg.Button('Microsoft Sudoku'), sg.Button('图片'), ],
          [sg.Image(img2, key='sudoku_img', size=(400, 300))],
          [sg.Text('Enter something on Row 2'), sg.InputText()],
          [sg.Button('识别'), sg.Button('求解'), ],
          [sg.Button('play', size=(5, 2)),
           sg.Slider((1, 10), key='count_play', orientation='horizontal', font=('Helvetica', 12))],
          # [sg.Multiline(key='info', size=(45, 9))],
          # [sg.Output(size=(80, 20))],
          [sg.Button('Clear'), sg.Button('Ok'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('Yuki - Sudoku 1.0', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break
    if event == 'Microsoft Sudoku':

        image = sudoku.grab_win()
        if sudoku.find_sudoku(image):
            size = (300, 300)
            image = cv.resize(sudoku.sudoku_img, size)
            window['sudoku_img'].update(img2bytes(image))

    if event == '识别':
        sudoku.split_recognized()
        # window['info'].update(sudoku.sudoku_data)

    if event == '求解':
        solver = SudokuSolver(sudoku.sudoku_data.copy())
        solve = solver.s()
        print(solve)
        print(solver.check())

    if event == 'play':
        count = int(values['count_play'])
        play(count)
    if event == 'Clear':
        window['output'].update('')
    print('You entered ', values)

window.close()
