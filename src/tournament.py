"""
The idea is to let our engine play a number of games against a reference engine.
We can export the games to a PGN file and then asses the ELO with Ordo.
"""

import chess
import chess.pgn

EXPORT_FOLDER = "games"

class Tournament:
    def __init__(self, engine, reference_engine, num_games=100):
        self.engine = engine
        self.reference_engine = reference_engine
        self.num_games = num_games
    
    def play(self):
        games = []
        for i in range(self.num_games):
            print(f"Playing game {i+1} of {self.num_games}")
            if i % 2 == 0:
                game = self.play_game(self.reference_engine, self.engine)
            else:
                game = self.play_game(self.engine, self.reference_engine)
            
            games.append(str(game))
            print(game)
            

        with open(f"{EXPORT_FOLDER}/games.pgn", "w") as f:
            f.write('\n\n'.join(games))

    def play_game(self, engine_white, engine_black):
        game = chess.pgn.Game()
        game.headers["White"] = engine_white.name
        game.headers["Black"] = engine_black.name
        node = game

        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = engine_white.play(board)
            else:
                move = engine_black.play(board)

            board.push(move)
            node = node.add_variation(move)

        game.headers["Result"] = board.result()

        return game

if __name__ == "__main__":
    from src.engines.stockfish import StockfishInterface
    from src.engines.neural import NeuralInterface

    e1 = StockfishInterface("src/engines/stockfish_17", elo=2000, max_time=0.05)
    e2 = NeuralInterface(use_mateness=True, prevent_stalemate=True, prevent_repetition=False, use_syzgy=False)

    t = Tournament(e1, e2, num_games=1000)
    t.play()

    e1.quit()
    e2.quit()