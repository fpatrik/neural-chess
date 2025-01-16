import torch
import chess
import chess.svg
import copy

import numpy as np
import matplotlib.pyplot as plt

from src.train import DEFAULT_ARGS
from src.model import ChessTranformerModel
from src.data.board_representation import fen_to_board
from src.engines.syzygy import SyzygyInterface

MODEL_PATH = "models/default_epoch_5.pth"
SYZYGY_PATH = "syzygy"

N_BINS = DEFAULT_ARGS['n_bins']

def copy_and_move(board, move):
    new_board = board.copy()
    new_board.move_stack = copy.deepcopy(board.move_stack)
    new_board.push(move)
    return new_board

class NeuralInterface:

    def __init__(self, use_mateness=True, prevent_stalemate=True, prevent_repetition=True, use_syzgy=True):
        self.use_mateness = use_mateness
        self.prevent_stalemate = prevent_stalemate
        self.prevent_repetition = prevent_repetition

        self.name = f"Neural Interface (mateness {use_mateness}, stalemate {prevent_stalemate}, repetition {prevent_repetition}, syzygy {use_syzgy})"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE", self.device)

        self.model = ChessTranformerModel(
            **DEFAULT_ARGS['model_args']
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

        self.syzygy = SyzygyInterface(SYZYGY_PATH) if use_syzgy else None

    def play(self, board):
        legal_moves = list(board.legal_moves)

        # Optionally, if we find a move in the endgame tablebases, we play it.
        if self.syzygy is not None:
            dtzs = [self.syzygy.get_dtz(copy_and_move(board, move)) for move in legal_moves]
            

            if any(e is not None for e in dtzs):
                move_idx = np.nanargmax(np.array(dtzs, dtype=float))
                return legal_moves[move_idx]

        # Optionally we filter out moves that lead to stalemate or repetition.
        filtered_legal_moves = []
        for move in legal_moves:
            new_board = copy_and_move(board, move)

            if new_board.is_stalemate() and self.prevent_stalemate:
                continue

            if (new_board.can_claim_threefold_repetition() or new_board.can_claim_fifty_moves()) and self.prevent_repetition:
                continue

            filtered_legal_moves.append(move)
        
        if len(filtered_legal_moves) == 0:
            filtered_legal_moves = legal_moves

        board_encodings = []
        for move in filtered_legal_moves:
            new_board = copy_and_move(board, move)
            board_encodings.append(fen_to_board(new_board.fen()))

        board_inputs = torch.from_numpy(np.array(board_encodings)).float().to(self.device)

        win_prob_logits, mateness_logit = self.model(board_inputs)
        bin_values = torch.nn.functional.softmax(win_prob_logits, dim=1).cpu().detach().numpy()
        mateness = torch.sigmoid(mateness_logit).cpu().detach().numpy()

        bin_means = np.arange(1 / (2 * N_BINS), 1, 1 / N_BINS)
        expected_win_percentages = np.dot(bin_values, bin_means)
    
        if self.use_mateness:
            expected_win_percentages -= mateness.flatten()

        move_idx = np.argmin(expected_win_percentages) # Choose minimum from opponent's expected win percentage.

        return filtered_legal_moves[move_idx]
    
    def quit(self):
        pass

if __name__ == "__main__":
    from src.engines.stockfish import StockfishInterface

    e_1 = StockfishInterface("src/engines/stockfish_17", elo=2400, max_time=0.05)
    e_2 = NeuralInterface()

    ply = 0
    board = chess.Board("3q2k1/1p3p2/p7/1N1p2Pp/3P2P1/1P1n1b2/r7/6K1 b - - 1 34")
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = e_1.play(board)
        else:
            move = e_2.play(board)

        board.push(move)
        print(board)
        print("---")

        board_svg = chess.svg.board(board=board)
        with open(f"plots/match/ply_{ply}.svg", "w") as f:
            f.write(board_svg)

        ply += 1

    print(board.outcome())
    e_2.quit()
    
    board_svg = chess.svg.board(board=board)
    with open("plots/end_position.svg", "w") as f:
        f.write(board_svg)