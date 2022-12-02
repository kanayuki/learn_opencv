import time
from datetime import datetime

import numpy as np
import pyautogui
import win32gui

from sudoku import Sudoku
from sudoku_solver import SudokuSolver


def play(n=1):
    print(f'start - {n}')
    handle = win32gui.FindWindow(None, 'Microsoft Sudoku')
    win32gui.ShowWindow(handle, 4)
    win32gui.SetForegroundWindow(handle)

    # 判断是否加载中
    count = 0
    while True:
        # pyautogui.moveTo(1, 1)
        play_again = pyautogui.locateCenterOnScreen('btn_img/play_again.png', confidence=0.8)
        if play_again:
            pyautogui.click(*play_again)

        time.sleep(0.2)
        if pyautogui.locateCenterOnScreen('btn_img/notes.png', confidence=0.8):
            count += 1

        if count == 2:
            break

    print('Sudoku loaded!')
    time.sleep(0.3)

    sudoku = Sudoku()
    sudoku.find_sudoku(sudoku.grab_win())
    sudoku.save_img('screenshot.jpg')
    sudoku.split_recognized()
    np.savetxt('data/sudoku {}.txt'.format(datetime.now().strftime('%Y%m%d %H%M%S')), sudoku.sudoku_data, '%d')

    solver = SudokuSolver(sudoku.sudoku_data.copy())
    solve = solver.s()
    print(solve)
    print(solver.check())

    # 自动填入
    rx = sudoku.bbox[0] + sudoku.clip_region[0]
    ry = sudoku.bbox[1] + sudoku.clip_region[1]
    pos_blank = rx + sudoku.xstep * 10, ry + sudoku.ystep / 2
    print(pos_blank)
    # time.sleep(0.1)
    pyautogui.click(*pos_blank)
    # time.sleep(0.1)

    data = solve - sudoku.sudoku_data
    print(f'fill: \n{data}')
    for num in range(1, 10):
        poss = np.argwhere(data == num)
        if poss.any():
            btn_num = pyautogui.locateCenterOnScreen(f'btn_img/btn_{num}.png', confidence=0.8)
            pyautogui.click(*btn_num)
            time.sleep(0.1)
            for i, j in poss:
                pos = rx + (j + 0.5) * sudoku.xstep, ry + (i + 0.5) * sudoku.ystep
                pyautogui.click(*pos)
                # time.sleep(0.1)

    # pyautogui.click(*pos_blank)
    # time.sleep(0.1)
    if n == 1:
        print('end!')
        return

    time.sleep(1)
    play(n - 1)


if __name__ == '__main__':
    play(3)
