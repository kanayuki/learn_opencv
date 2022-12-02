import cv2
import numpy as np


class SudokuDrawer:
    def __init__(self, *args):
        self.size = 300
        self.canvas = np.full((self.size, self.size, 3), 255, np.uint8)
        self.sudoku_data = None
        self.step = self.size / 9

    def set_size(self, size):
        self.size = size
        self.step = size / 9

    def draw(self):
        color = 0
        thickness = 2
        s = self.size
        img = self.canvas
        cv2.rectangle(img, (0, 0), (s, s), color, thickness)
        cv2.line(img, (0, int(s / 3)), (s, int(s / 3)), color, thickness)
        cv2.line(img, (0, int(s * 2 / 3)), (s, int(s * 2 / 3)), color, thickness)

        cv2.line(img, (int(s / 3), 0), (int(s / 3), s), color, thickness)
        cv2.line(img, (int(s * 2 / 3), 0), (int(s * 2 / 3), s), color, thickness)

        for i in range(9):
            for j in range(9):
                n = self.sudoku_data[i, j]
                if n == 0:
                    continue
                self.fill(i, j, n)
        cv2.imshow('sudoku', img)

        # def onMouse(events,x,y,flags):
        #     if events==cv2.EVENT_LBUTTONDOWN:
        #
        #
        # cv2.setMouseCallback('sudoku',onMouse)
        cv2.waitKey()

    def fill(self, i, j, n, color=255):
        org = int((i + 0.1) * self.step), int((j + 0.8) * self.step)
        cv2.putText(self.canvas, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


if __name__ == '__main__':
    drawer = SudokuDrawer()
    drawer.sudoku_data = np.loadtxt('data/sudoku 20221201065056.txt', np.uint8)

    drawer.draw()
