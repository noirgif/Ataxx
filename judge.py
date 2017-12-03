import numpy as np

class ChessBoard:
    """The Ataxx ChessBoard

    board: a SIZExSIZE(by default 7) array,
     each element is an integer which indicate the state of the place
    """
    SIZE = 7
    def __init__(self):
        self.board = np.zeros((ChessBoard.SIZE, ChessBoard.SIZE), dtype=np.int8)
        self.board[0][0] = 1
        self.board[0][ChessBoard.SIZE - 1] = -1
        self.board[ChessBoard.SIZE - 1][0] = -1
        self.board[ChessBoard.SIZE - 1][ChessBoard.SIZE - 1] = 1

    def have(self, point=(0, 0), shape=(1, 1)):
        """check if there's a 1 piece in the `shape` around the `point` given"""
        # subboard is the range around the point, within the board
        subboard = self.board[\
            max(0, point[0] - shape[0]):min(ChessBoard.SIZE, point[0] + shape[0] + 1),\
            max(0, point[1] - shape[1]):min(ChessBoard.SIZE, point[1] + shape[1] + 1)]
        return (subboard == 1).any()

    def reverse(self):
        """reverse the board, exchange the two colors of the pieces"""
        self.board = -self.board

    def put(self, source=(0, 0), dest=(0, 0)):
        """put a piece of color `chess` on the `point`
            return: True successful, False failed"""
        # don't permit out-of-range moves
        # may be optimized later
        if min(source) < 0 or max(source) > ChessBoard.SIZE or \
            min(dest) < 0 or max(dest) > ChessBoard.SIZE:
            return False
        # must move a piece of oneself
        if self.board[source] != 1:
            return False
        # cannot move onto another piece
        if self.board[dest] != 0:
            return False
        # the distance of the move
        dist = max([abs(source[i] - dest[i]) for i in range(2)])
        if dist > 2:
            return False
        elif dist == 2:
            # jump
            self.board[source] = 0
            self.board[dest] = 1
        else:
            # clone
            self.board[dest] = 1
        # change neighbors' color
        self.board[max(0, dest[0] - 1):min(ChessBoard.SIZE, dest[0] + 2),\
            max(0, dest[1] - 1):min(ChessBoard.SIZE, dest[1] + 2)] &= 1
        return True

    def __repr__(self):
        return self.board.__repr__()


if __name__ == '__main__':
    chessboard = ChessBoard()
    print(chessboard)
