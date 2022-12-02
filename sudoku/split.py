import cv2 as cv
import numpy as np

from microsoft_sudoku import MicrosoftSudoku
from sudoku_cnn import recognizer


def sudoku_clip(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if not ret:
        raise Exception('二值化失败')
    # print(bin_img)
    gray = cv.bitwise_not(bin_img)
    canny = cv.Canny(gray, 50, 150)
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    curves = [cont for cont in contours if cv.contourArea(cont) > 4e4]
    if curves:
        poly = cv.approxPolyDP(curves[0], 10, True)

        # 裁剪出数独区域
        x_min, y_min = np.min(np.squeeze(poly), 0)
        x_max, y_max = np.max(np.squeeze(poly), 0)
        return gray[y_min:y_max, x_min: x_max]


def sudoku_split(sudoku_img, size):
    h, w = sudoku_img.shape
    x_step, y_step = int(w / 9), int(h / 9)
    print('step:', x_step, y_step)
    tiles = []

    for i in range(9):
        for j in range(9):
            tile = sudoku_img[i * y_step:(i + 1) * y_step, j * x_step:(j + 1) * x_step]
            e = int(x_step * 0.12)
            tile = tile[e:-e, e:-e]
            tile = cv.GaussianBlur(tile, (5, 5), 0)
            tiles.append(tile)
    return tiles


def pad(img):
    r, c = img.shape
    col = np.full((r, 1), 255).astype(np.uint8)
    img = np.hstack([col, img, col])
    row = np.full((1, c + 2), 255, np.uint8)
    return np.vstack([row, img, row])


def combine(imgs):
    imgs = [pad(i) for i in imgs]
    imgs = np.hstack(imgs)
    return np.vstack(np.hsplit(imgs, 9))


def save2dataset(imgs, order):
    recognize = recognizer()
    for image in imgs:
        if cv.countNonZero(image) == 0:
            continue
        image = cv.resize(image, (30, 30))
        order += 1
        cv.imwrite('../Image/nums/{}-num-{}.png'.format(order, recognize(image)), image)


if __name__ == '__main__':
    microsoft_sudoku = MicrosoftSudoku()
    shot = microsoft_sudoku.snapshot()
    shot = cv.cvtColor(np.array(shot), cv.COLOR_RGB2BGR)
    # cv.imwrite("../Image/sudoku_001_tiles.png", tiles)

    sudoku_img = sudoku_clip(shot)
    tiles = sudoku_split(sudoku_img, 30)

    # save2dataset(tiles, 222)
    new_img = combine(tiles)

    cv.imshow("tiles", shot)
    cv.imshow("sudoku_img", sudoku_img)
    cv.imshow("new_img", new_img)

    key = cv.waitKey()
    if key == ord('q'):
        print(key)
