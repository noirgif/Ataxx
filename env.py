import numpy as np
from copy import deepcopy

SIZE = 7


def new_board():
    board = np.zeros(
        (SIZE, SIZE), dtype=np.int32)
    board[0][0] = 1
    board[0][SIZE - 1] = -1
    board[SIZE - 1][0] = -1
    board[SIZE - 1][SIZE - 1] = 1
    return board


def full(board):
    """check if the board is full"""
    if board.all():
        return True
    else:
        return False


def have(board, point=(0, 0), shape=(1, 1)):
    """check if there's a 1 piece in the `shape` around the `point` given"""
    # subboard is the range around the point, within the board
    subboard = board[
        max(0, point[0] - shape[0]):min(SIZE, point[0] + shape[0] + 1),
        max(0, point[1] - shape[1]):min(SIZE, point[1] + shape[1] + 1)]
    return (subboard == 1).any()


def put(board, move):
    """move a piece on the board
        move: a tuple being (sourcex, sourcey, destx, desty)
        returns: a tuple of (bool, board)

            bool: True successful, False failed

            board: the change on the board"""
    # must move a piece of oneself
    if board[move[0], move[1]] != 1:
        return False, None
    # cannot move onto another piece
    if board[move[2], move[3]] != 0:
        return False, None
    # the distance of the move
    dist = max([abs(move[i] - move[2 + i]) for i in range(2)])
    if dist > 2:
        return False, None
    elif dist == 2:
        # jump
        board[move[0], move[1]] = 0
        board[move[2], move[3]] = 1
    else:
        # clone
        board[move[0], move[1]] = 1
    # change neighbors' color
    board[max(0, move[2] - 1):min(SIZE, move[2] + 2),
               max(0, move[3] - 1):min(SIZE, move[3] + 2)] &= 1
    return True, board


def get_moves(board):
    """get the moves available in current board"""
    moves = []
    for x in range(SIZE):
        for y in range(SIZE):
            if bool(board[x, y] == 1):
                for dx in range(x - 2, x + 3):
                    for dy in range(y - 2, y + 3):
                        if (dx in range(SIZE)) and (dy in range(SIZE)):
                            if board[dx, dy] == 0:
                                result = put(board, (x, y, dx, dy))
                                if not result[0]:
                                    print("Error when applying", (x, y,
                                                                  dx, dy), "to", board)
                                else:
                                    yield x[1]
    yield np.zeros((SIZE, SIZE), dtype=np.int8)


def step(board, action):
    """make a step in current board
        return : a tuple of reward, done"""
    board += action
    # the reward is the total pieces conquered
    return float(action.sum()), bool(board.all())
