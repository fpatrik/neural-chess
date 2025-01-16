import chess
import pandas as pd

from tqdm import tqdm
from src.engines.neural import NeuralInterface
from src.engines.stockfish import StockfishInterface
from src.train import DEFAULT_ARGS

PUZZLE_PATH = 'data/puzzles.csv'

if __name__ == '__main__':
    df = pd.read_csv(PUZZLE_PATH)
    engine = NeuralInterface(use_mateness=True, use_syzgy=False)
    #engine = StockfishInterface("src/engines/stockfish_17", max_time=0.1)

    correct = 0
    total = 0
    for i, row in tqdm(df.iterrows(), desc="Evaluating puzzles", unit=" puzzles"):
        fen = row['FEN']
        board = chess.Board(fen)
        
        correct_moves = row['Solution'].split(' ')

        is_engine_move = False
        mistake = False
        for correct_move in correct_moves:
            expected_move = board.parse_san(correct_move)
            if is_engine_move:
                engine_move = engine.play(board)

                if expected_move != engine_move:
                    # Sometimes there are more than one checkmate option
                    board.push(engine_move)
                    if not board.is_checkmate():
                        mistake = True
                    break

            is_engine_move = not is_engine_move
            board.push(expected_move)
        
        if not mistake:
            correct += 1
        total += 1

    engine.quit()

    print(f"{correct} correct out of {total} ({correct / total * 100:.2f}%)")
