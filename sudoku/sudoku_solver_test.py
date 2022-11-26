import numpy as np

sudoku_data = np.loadtxt('sudoku_data.txt', np.int32)
print(sudoku_data)


def block(x, y):
    i, j = np.floor([x / 3, y / 3]).astype(int)
    return sudoku_data[i * 3:i * 3 + 3, j * 3:j * 3 + 3]


def row(n):
    return sudoku_data[n, :]


def col(n):
    return sudoku_data[:, n]


def may(x, y):
    sub = set(block(x, y).ravel())
    return nums - sub - set(row(x)) - set(col(y))


nums = set(range(1, 10))

solve_data = sudoku_data


def solve():
    for i in range(9):
        for j in range(9):
            n = solve_data[i, j]
            if n == 0:  # 未填
                n_may = may(i, j)
                print('{}-{}-{}'.format(i + 1, j + 1, n_may))
                if len(n_may) == 1:  # 唯一确定
                    solve_data[i, j] = list(n_may)[0]


if __name__ == '__main__':
    # print(block(6, 8))
    solve()
    print(solve_data)
    solve()
    print(solve_data)
    solve()
    print(solve_data)
