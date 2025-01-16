import torch
import chess
import chess.svg

import numpy as np
import matplotlib.pyplot as plt

from src.train import DEFAULT_ARGS
from src.model import ChessTranformerModel
from src.data.board_representation import fen_to_board

POSITION_FEN = "4q1k1/1p3p2/p7/1N1p2Pp/3P2P1/1P1n1b2/r7/6K1 w - - 2 31"
MODEL_PATH = "models/default_epoch_300.pth"
N_BINS = DEFAULT_ARGS['n_bins']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", device)

model = ChessTranformerModel(
    **DEFAULT_ARGS['model_args']
)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
model.to(device)

board_encoding = fen_to_board(POSITION_FEN)
board_input = torch.from_numpy(board_encoding).float().unsqueeze(0).to(device)

win_probs, mateness = model(board_input)

bin_values = torch.nn.functional.softmax(win_probs, dim=1).cpu().detach().numpy()[0]

bin_means = np.arange(1 / (2 * N_BINS), 1, 1 / N_BINS)
expected_win_percentages = np.dot(bin_values, bin_means)

mateness_value = torch.nn.functional.sigmoid(mateness).cpu().detach().numpy()[0]

print(f"Expected win probability: {expected_win_percentages}")
print(f"Mateness: {mateness_value}")

x = np.arange(1 / (2 * N_BINS), 1, 1 / N_BINS)
plt.bar(x, bin_values, width=1 / N_BINS)
plt.xlabel("Win Probability")
plt.ylabel("Density")
plt.title("Predicted Win Probability Distribution")
plt.grid()
plt.savefig("plots/predicted_distribution.png")
plt.clf()

board_svg = chess.svg.board(board=chess.Board(POSITION_FEN))
with open("plots/evaluated_position.svg", "w") as f:
    f.write(board_svg)