import numpy as np


class SudokuSolver:
    def __init__(self, sudoku_data):
        self.nums = set(range(1, 10))

        self.solve_data = sudoku_data

        self.unknown_list = []
        for i in range(9):
            for j in range(9):
                n = self.solve_data[i, j]
                if n == 0:  # 未填
                    self.unknown_list.append((i, j))

    def block(self, x, y):
        i, j = np.floor([x / 3, y / 3]).astype(int)
        return self.solve_data[i * 3:i * 3 + 3, j * 3:j * 3 + 3]

    def row(self, n):
        return self.solve_data[n, :]

    def col(self, n):
        return self.solve_data[:, n]

    def may(self, x, y):
        sub = self.block(x, y).ravel()
        return self.nums - set(sub) - set(self.row(x)) - set(self.col(y))

    def solve(self):
        solved = False
        count = 0
        while not solved:
            count += 1
            new_unknown_list = []
            for i, j in self.unknown_list:
                n_may = self.may(i, j)
                print('{}-{}-{}'.format(i + 1, j + 1, n_may))
                if len(n_may) == 1:  # 唯一确定
                    self.solve_data[i, j] = list(n_may)[0]
                else:
                    new_unknown_list.append((i, j))
            self.unknown_list = new_unknown_list
            if len(self.unknown_list) == 0:
                solved = True
            print(f'第{count}次求解{self.solve_data}')

        return self.solve_data


if __name__ == '__main__':
    sudoku_data = np.loadtxt('sudoku_data.txt', np.int32)
    print(sudoku_data)
    solve = SudokuSolver(sudoku_data).solve()

    print(solve)




