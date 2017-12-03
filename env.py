import numpy as np

__all__ = ["get", "step"]

class ChessBoard:
    """The Ataxx ChessBoard

    board: a SIZExSIZE(by default 7) array,
     each element is an integer which indicate the state of the place
    """
    SIZE = 7

    def __init__(self):
        self.board = np.zeros(
            (ChessBoard.SIZE, ChessBoard.SIZE), dtype=np.int32)
        self.board[0][0] = 1
        self.board[0][ChessBoard.SIZE - 1] = -1
        self.board[ChessBoard.SIZE - 1][0] = -1
        self.board[ChessBoard.SIZE - 1][ChessBoard.SIZE - 1] = 1
        self.turns = 0

    def full(self):
        """check if the board is full"""
        if self.board.all():
            return True
        else:
            return False

    def have(self, point=(0, 0), shape=(1, 1)):
        """check if there's a 1 piece in the `shape` around the `point` given"""
        # subboard is the range around the point, within the board
        subboard = self.board[
            max(0, point[0] - shape[0]):min(ChessBoard.SIZE, point[0] + shape[0] + 1),
            max(0, point[1] - shape[1]):min(ChessBoard.SIZE, point[1] + shape[1] + 1)]
        return (subboard == 1).any()

    def reverse(self):
        """reverse the board, exchange the two colors of the pieces"""
        self.board = -self.board

    def put(self, move):
        """put a piece of color 1 on the `point`
            return: True successful, False failed"""
        
        # must move a piece of oneself
        if self.board[source[0], source[1]] != 1:
            return False
        # cannot move onto another piece
        if self.board[dest[0], source[1]] != 0:
            return False
        # the distance of the move
        dist = max([abs(source[i] - dest[i]) for i in range(2)])
        if dist > 2:
            return False
        elif dist == 2:
            # jump
            self.board[source[0], source[1]] = 0
            self.board[dest[0], source[1]] = 1
        else:
            # clone
            self.board[dest[0], source[1]] = 1
        # change neighbors' color
        self.board[max(0, dest[0] - 1):min(ChessBoard.SIZE, dest[0] + 2),
                   max(0, dest[1] - 1):min(ChessBoard.SIZE, dest[1] + 2)] &= 1
        return True

    def score(self):
        """return the score of current play, maybe it's a good reward"""
        return self.board.sum()

    def __repr__(self):
        return self.board.__repr__()


myChessBoard = None

def step(action):
    """make a step in current board
        return : a tuple of reward, done"""
    global myChessBoard
    if not action.size:
        print("Pass")
    else:
        result = myChessBoard.put(action[:2], action[2:])
    myChessBoard.reverse()
    myChessBoard.turns += 1
    if result:
        return float(-myChessBoard.score()), myChessBoard.full()
    else:
        # a failing move
        return float(-myChessBoard.score() - 10000), True


def reset():
    """reset the environment"""
    global myChessBoard
    print("New comp")
    myChessBoard = ChessBoard()


def get():
    global myChessBoard
    return myChessBoard.board
