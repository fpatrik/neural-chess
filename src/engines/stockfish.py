import chess
import chess.engine

STOCKFISH_PATH = "src/engines/stockfish_17"

class StockfishInterface:
    def __init__(self, path, elo=None, max_time=0.5):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.elo = elo
        self.max_time = max_time

        if self.elo is not None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        
        self.name = f"Stockfish {self.elo}" if self.elo is not None else "Stockfish"

    def play(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=self.max_time))
        return result.move

    def quit(self):
        self.engine.quit()

if __name__ == "__main__":
    sf_1 = StockfishInterface(STOCKFISH_PATH, elo=1500)
    sf_2 = StockfishInterface(STOCKFISH_PATH, elo=2500)

    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = sf_1.play(board)
        else:
            move = sf_2.play(board)

        board.push(move)
        print(board)

    sf_1.quit()
    sf_2.quit()

"""

# Check available options.
print(engine.options["UCI_LimitStrength"])
print(engine.options["UCI_Elo"])
# Option(name='Hash', type='spin', default=16, min=1, max=131072, var=[])

# Set an option.
#engine.configure({"Hash": 32})
engine.quit()
"""