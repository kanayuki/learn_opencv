import win32gui
from PIL import ImageGrab


class MicrosoftSudoku:
    def __init__(self):
        self.title = 'Microsoft Sudoku'
        self.handle = win32gui.FindWindow(None, self.title)
        self.bbox = None
        if self.handle is None:
            raise Exception('{} 未启动'.format(self.handle))

    def show(self):
        handle = self.handle
        win32gui.ShowWindow(handle, 4)
        win32gui.SetForegroundWindow(handle)

    def snapshot(self):
        self.show()
        x1, y1, x2, y2 = win32gui.GetWindowRect(self.handle)
        self.bbox = [x1, y1, x2 - x1, y2 - y1]
        return ImageGrab.grab(self.bbox)
