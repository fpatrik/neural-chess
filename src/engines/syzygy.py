import chess.syzygy

class SyzygyInterface:
    def __init__(self, path: str):
        self.path = path
    
    def get_dtz(self, board):
        # Only positions with 5 or fewer pieces are supported
        if board.occupied.bit_count() > 5:
            return None

        with chess.syzygy.open_tablebase(self.path) as tablebase:
            dtz = tablebase.get_dtz(board)
        
        return dtz


if __name__ == '__main__':
    syzygy_interface = SyzygyInterface("syzygy")

    board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")

    print(syzygy_interface.get_dtz(board))