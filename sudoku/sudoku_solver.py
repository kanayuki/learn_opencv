import time

import numpy as np


class SudokuSolver:
    def __init__(self, sudoku_data):
        self.nums = set(range(1, 10))

        self.solve_data = sudoku_data

        self.unknown_list = []
        self.num_pos = [[] for i in range(9)]
        self.mask = [np.full((9, 9), False, bool) for i in range(9)]
        self.count = 0
        self.not_fill = [list(set() for i in range(9)) for j in range(9)]
        for i in range(9):
            for j in range(9):
                n = self.solve_data[i, j]
                if n == 0:  # 未填
                    self.unknown_list.append((i, j))
                else:
                    self.num_pos[n - 1].append((i, j))
                    self.set_mask(n, i, j)

    def index_block(self, x, y) -> np.ndarray:
        return np.floor([x / 3, y / 3]).astype(int)

    def block(self, x, y) -> np.ndarray:
        i, j = np.floor([x / 3, y / 3]).astype(int)
        return self.solve_data[i * 3:i * 3 + 3, j * 3:j * 3 + 3]

    def row(self, n) -> np.ndarray:
        return self.solve_data[n, :]

    def col(self, n):
        return self.solve_data[:, n]

    def may(self, x, y):
        sub = self.block(x, y).ravel()
        return self.nums - set(sub) - set(self.row(x)) - set(self.col(y)) - self.not_fill[x][y]

    def fill(self, x, y, n):
        self.solve_data[x, y] = n
        self.set_mask(n, x, y)
        self.unknown_list.remove((x, y))

    def set_mask(self, n, x, y):
        self.mask[n - 1][x, :] = True  # row
        self.mask[n - 1][:, y] = True  # col
        # x, y = np.floor([x / 3, y / 3]).astype(int)
        x, y = self.index_block(x, y)
        self.mask[n - 1][x * 3:x * 3 + 3, y * 3:y * 3 + 3] = True  # block

    def check(self):
        sum_col = np.sum(self.solve_data, 0)
        sum_row = np.sum(self.solve_data, 1)
        if np.all(sum_col == 45) and np.all(sum_row == 45):
            return True

    def solve(self):
        while True:
            self.count += 1
            sl = len(self.unknown_list)

            # mask
            for ind in range(9):
                new_mask = (self.solve_data > 0) | self.mask[ind]
                for i in range(3):
                    for j in range(3):
                        sub = new_mask[i * 3:i * 3 + 3, j * 3:j * 3 + 3]
                        if sub.all() and ((ind + 1) not in self.block(i * 3, j * 3)):  # 假定错误
                            print(f'第 {self.count} 次求解: [{i + 1}, {j + 1}] 宫无法填写数字 {ind + 1}')
                            return False
                        values, frequency = np.unique(sub, return_counts=True)
                        if (not values[0]) and (frequency[0] == 1):  # 唯一确定
                            x, y = np.argwhere(np.bitwise_not(sub))[0]  # 相对索引
                            x, y = i * 3 + x, j * 3 + y
                            self.fill(x, y, ind + 1)

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
        copy_unknown = self.unknown_list[:]
        for x, y in copy_unknown:
            self.solve_data = np.copy(copy)  # 假定初始状态
            self.mask = np.copy(copy_mask)
            self.unknown_list = copy_unknown[:]
            n_may = self.may(x, y)
            print('[{}, {}] 处可能的值有：{} '.format(x + 1, y + 1, n_may))
            for v in n_may:
                self.solve_data = np.copy(copy)  # 假定初始状态
                self.mask = np.copy(copy_mask)
                self.unknown_list = copy_unknown[:]

                print('假定 [{}, {}] 处为 {}'.format(x + 1, y + 1, v))
                self.fill(x, y, v)
                res = self.solve()
                if res:
                    return self.solve_data
                elif res is False:
                    self.not_fill[x][y].add(v)
                    print('拒绝矩阵:\n{}'.format(self.not_fill))

        raise Exception('求解失败！')


if __name__ == '__main__':
    sudoku_data = np.loadtxt('data/sudoku 20221201015640.txt', np.int32)
    # sudoku_data = np.load('data/sudoku_data-3.npy')
    print(sudoku_data)
    start = time.time()
    solver = SudokuSolver(sudoku_data)
    solve = solver.s()
    print('花费时间：{}'.format(time.time() - start))
    print(solve)
    print(solver.check())
