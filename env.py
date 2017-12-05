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


def put(old_board, move):
    """move a piece on the board
        move: a tuple being (sourcex, sourcey, destx, desty)
        returns: a tuple of (bool, board)

            bool: True successful, False failed

            board: the change on the board"""
    board = deepcopy(old_board)
    # the distance of the move
    dist = max([abs(move[i] - move[2 + i]) for i in range(2)])
    if dist == 2:
        # jump
        board[move[0], move[1]] = 0
        board[move[2], move[3]] = 1
    else:
        # clone
        board[move[2], move[3]] = 1
    # change neighbors' color
    board[max(0, move[2] - 1):min(SIZE, move[2] + 2),
          max(0, move[3] - 1):min(SIZE, move[3] + 2)] &= 1
    return True, board - old_board


dx1 = list(range(-1, 2))
dx1 = dx1 + [-1, 1] + dx1
dy1 = ([-1] * 3) + ([0] * 2) + ([1] * 3)
d1 = zip(dx1, dy1)
dx2 = list(range(-2, 3))
dx2 = dx2 + ([-2, 2] * 3) + dx2
dy2 = ([-2] * 5) + [-1, -1, 0, 0, 1, 1] + ([2] * 5)
d2 = zip(dx2, dy2)
av_moves = list(d1) + list(d2)


def get_moves(board):
    """get the moves available in current board"""
    moves = []
    clone_found = False
    no_move = True
    for x in range(SIZE):
        for y in range(SIZE):
            if (board[x, y] == 0).any():
                for dx, dy in av_moves:
                    xx = x + dx
                    yy = y + dy
                    if xx in range(SIZE) and yy in range(SIZE)\
                            and (board[xx, yy] == 1).any():
                        if (dx, dy) in d1:
                            if clone_found:
                                continue
                            else:
                                clone_found = True
                        no_move = False
                        yield put(board, (xx, yy, x, y))[1]
    if no_move:
        yield np.zeros((SIZE, SIZE), dtype=np.int32)


def step(board, action):
    """make a step in current board
        return : a tuple of reward, done"""
    board += action
    # the reward is the total pieces conquered
    return float(action.sum()), not bool((board == 1).any() and (board == -1).any() and not board.all())


class Play:
    """wrapping of an Ataxx play"""

    def __init__(self):
        self.b = new_board()

    def reset(self):
        """ reset the chessboard"""
        self.b = new_board()

    def step(self, action):
        """ make a step in current board

            returns a tuple of reward, done"""
        reward, done = step(self.b, action)
        if done:
            reward += 765 if (self.b == 1).any() else -765
        # reverse the board
        self.b = -self.b
        return reward, done
