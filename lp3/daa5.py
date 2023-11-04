def is_safe(board, row, col):
    for i in range(row):
        if board[i][col] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, len(board))):
        if board[i][j] == 1:
            return False
    return True

def solve_n_queens(board, row):
    n = len(board)
    if row == n:
        return True

    for col in range(n):
        if is_safe(board, row, col):
            board[row][col] = 1

            if solve_n_queens(board, row + 1):
                return True

            board[row][col] = 0

    return False

def print_board(board):
    for row in board:
        print(' '.join(['Q' if x == 1 else '.' for x in row]))

n = 8
chessboard = [[0] * n for _ in range(n)]


chessboard[0][3] = 1
if solve_n_queens(chessboard, 1):
    print("Solution found:")
    print_board(chessboard)
else:
    print("No solution found.")