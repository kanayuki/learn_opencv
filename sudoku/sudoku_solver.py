import os
import time

import numpy as np


class SudokuSolver:
    def __init__(self, sudoku_data):
        self.nums = set(range(1, 10))
        self.count = 0

        self.solve_data = sudoku_data

        self.unknown_list = []
        self.mask = np.full((9, 9, 9), False, bool)
        self.sub = np.full((3, 3, 9), False, bool)
        self.not_fill = np.full((9, 9, 9), False, bool)
        for i in range(9):
            for j in range(9):
                n = self.solve_data[i, j]
                if n == 0:  # 未填
                    self.unknown_list.append((i, j))
                else:
                    self.set_mask(i, j, n)
                    self.set_sub(i, j, n)

    def block(self, x, y) -> np.ndarray:
        i, j = np.floor([x / 3, y / 3]).astype(int)
        return self.solve_data[i * 3:i * 3 + 3, j * 3:j * 3 + 3]

    def may(self, x, y):
        sub = set(self.block(x, y).ravel())
        row = set(self.solve_data[x, :])
        col = set(self.solve_data[:, y])
        not_full = set(np.flatnonzero(self.not_fill[x, y]) + 1)
        return self.nums - sub - row - col - not_full

    def fill(self, x, y, n):
        self.solve_data[x, y] = n
        self.set_mask(x, y, n)
        self.set_sub(x, y, n)
        self.unknown_list.remove((x, y))

    def get_mask(self, n):
        return (self.solve_data > 0) | self.mask[:, :, n - 1]

    def set_mask(self, x, y, n):
        self.mask[x, :, n - 1] = True  # row
        self.mask[:, y, n - 1] = True  # col
        x, y = int(x / 3), int(y / 3)
        # x, y = self.index_block(x, y)
        self.mask[x * 3:(x + 1) * 3, y * 3:(y + 1) * 3, n - 1] = True  # block

    def set_sub(self, x, y, n):
        self.sub[int(x / 3), int(y / 3), n - 1] = True

    def check(self):
        sum_col = np.sum(self.solve_data, 0)
        sum_row = np.sum(self.solve_data, 1)
        if np.all(sum_col == 45) and np.all(sum_row == 45):
            return True

    def solve(self):
        while True:
            self.count += 1
            sl = len(self.unknown_list)

            # row-col
            new_unknown_list = self.unknown_list[:]
            for i, j in new_unknown_list:
                n_may = self.may(i, j)
                # print('{}-{}-{}'.format(i + 1, j + 1, n_may))
                ln = len(n_may)
                if ln == 0:  # 假定错误
                    print(f'第 {self.count} 次求解: [{i + 1}, {j + 1}] 处为空且无法填写')
                    return False
                if ln == 1:  # 唯一确定
                    n = list(n_may)[0]
                    self.fill(i, j, n)
            # mask
            for i in range(3):
                for j in range(3):
                    for n in range(1, 10):
                        if self.sub[i, j, n - 1]:
                            continue
                        sub_mask = self.get_mask(n)[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                        if sub_mask.all():  # 假定错误
                            print(f'第 {self.count} 次求解: [{i + 1}, {j + 1}] 宫无法填写数字 {n}')
                            return False
                        ind = np.argwhere(np.bitwise_not(sub_mask))
                        if len(ind) == 1:  # 唯一确定
                            x, y = ind[0]  # 相对索引
                            x, y = i * 3 + x, j * 3 + y
                            self.fill(x, y, n)

            if len(self.unknown_list) == sl:
                print(f'第 {self.count} 次求解: 剩余位置{sl}个')
                return None

            if len(self.unknown_list) == 0:
                print(f'第 {self.count} 次求解:\n{self.solve_data}')
                print('求解成功！')
                return True

            print(f'第 {self.count} 次求解:\n{self.solve_data}')

    def s(self):
        res = self.solve()
        if res:
            return self.solve_data

        print('开始尝试')
        copy = np.copy(self.solve_data)
        copy_mask = np.copy(self.mask)
        copy_sub = np.copy(self.sub)
        copy_unknown = self.unknown_list[:]
        for x, y in copy_unknown:
            self.solve_data = np.copy(copy)  # 假定初始状态
            self.mask = np.copy(copy_mask)
            self.sub = np.copy(copy_sub)
            self.unknown_list = copy_unknown[:]
            n_may = self.may(x, y)
            print('[{}, {}] 处可能的值有：{} '.format(x + 1, y + 1, n_may))
            for v in n_may:
                self.solve_data = np.copy(copy)  # 假定初始状态
                self.mask = np.copy(copy_mask)
                self.sub = np.copy(copy_sub)
                self.unknown_list = copy_unknown[:]

                print('假定 [{}, {}] 处为 {}'.format(x + 1, y + 1, v))
                self.fill(x, y, v)
                res = self.solve()
                if res:
                    return self.solve_data
                elif res is False:
                    self.not_fill[x, y, v - 1] = True
                    # print('拒绝矩阵:\n{}'.format(self.not_fill))

        raise Exception('求解失败！')


def test_mul():
    fs = os.listdir('data')
    ts = []
    for name in fs:
        if '.txt' not in name:
            continue

        path = os.path.join('data', name)
        sudoku_data = np.loadtxt(path, np.uint8)
        start = time.time()
        solver = SudokuSolver(sudoku_data)
        try:
            solver.s()
        except Exception as e:
            print(path)
            print(sudoku_data)

        spend = time.time() - start
        ts.append(spend)

    print('平均时间：{}'.format(np.mean(ts)))


def test():
    sudoku_data = np.loadtxt('data/sudoku 20221201060331.txt', np.int32)
    # sudoku_data = np.load('data/sudoku_data-3.npy')
    print(sudoku_data)
    start = time.time()
    solver = SudokuSolver(sudoku_data)
    solve = solver.s()
    print('花费时间：{}'.format(time.time() - start))
    print(solve)
    print(solver.check())


if __name__ == '__main__':
    # test()
    test_mul()
