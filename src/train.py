import torch
import json

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from torchinfo import summary

from src.data.bagz import BagDataSource
from src.data.dataset import ChessDataset
from src.model import ChessTranformerModel

DEFAULT_ARGS = {
    'name': 'default',
    'train_data_path': 'data/train_mate.bag',
    'test_data_path': 'data/test_mate.bag',
    'batch_size': 1024,
    'restrict_dataset_size': None,
    'samples_per_epoch': 10_000_000,
    'input_dim': 56,
    'n_bins': 128,
    'learning_rate': 3e-4,
    'epochs': 300,
    'mateness_importance': 1e1,
    'model_args': {
        'input_dim': 56,
        'model_dim': 256,
        'bin_dim': 128,
        'n_heads': 8,
        'num_layers': 6,
        'dim_feedforward': 512,
        'dropout': 0
    },
}

def train_model(args=DEFAULT_ARGS):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device)

    # Load the dataset
    train_bag_data_source = BagDataSource(args['train_data_path'])
    train_dataset = ChessDataset(train_bag_data_source, n_bins=args['n_bins'], restrict_dataset_size=args['restrict_dataset_size'])

    test_bag_data_source = BagDataSource(args['test_data_path'])
    test_dataset = ChessDataset(test_bag_data_source, n_bins=args['n_bins'])
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    # Initialize the model
    model = ChessTranformerModel(
        **args['model_args']
    )
    model_summary = summary(model, input_size=(args['batch_size'], 64, args['input_dim']))
    model.to(device)
    model.compile(mode="reduce-overhead")

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], fused=True)

    # Initialize the loss function
    win_prob_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    mateness_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    # Metrics
    mae = torch.nn.L1Loss(reduction="sum")

    # History
    rolling_train_loss_history = []
    rolling_train_loss_win_prob_history = []
    rolling_train_loss_mateness_history = []
    rolling_train_mae_history = []

    test_loss_history = []
    test_loss_win_prob_history = []
    test_loss_mateness_history = []
    test_mae_history = []

    len_test_data = len(test_dataset)

    # Train the model
    for epoch in range(args['epochs']):
        # Since the whole train dataset is very large at 500M+ samples, we use a random sampler to sample a subset of the data for every epoch
        random_sampler = RandomSampler(train_dataset, num_samples=args['samples_per_epoch'])
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], sampler=random_sampler, num_workers=8)

        model.train()

        rolling_train_loss = 0
        rolling_train_loss_win_prob = 0
        rolling_train_loss_mateness = 0
        rolling_train_mae = 0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}"):
            optimizer.zero_grad()

            board_representation, label, win_prob_bins, mateness = batch
            board_representation = board_representation.to(device)
            label = label.to(device)
            win_prob_bins = win_prob_bins.to(device)
            mateness = mateness.to(device)

            win_prob_logits, mateness_logit = model(board_representation)

            win_prob_loss = win_prob_loss_fn(win_prob_logits, win_prob_bins)
            mateness_loss = args['mateness_importance'] * mateness_loss_fn(mateness_logit, mateness)

            loss = win_prob_loss + mateness_loss

            loss.backward()
            optimizer.step()

            rolling_train_loss += loss.detach()
            rolling_train_loss_win_prob += win_prob_loss.detach()
            rolling_train_loss_mateness += mateness_loss.detach()

            # Compute the mode of the predicted distribution and compare it to the label
            mode = torch.argmax(win_prob_logits, dim=1)
            rolling_train_mae += mae(mode.detach() / args['n_bins'] + (1 / (2 * args['n_bins'])), label)

        rolling_train_loss_history.append(rolling_train_loss.item() / args['samples_per_epoch'])
        rolling_train_loss_win_prob_history.append(rolling_train_loss_win_prob.item() / args['samples_per_epoch'])
        rolling_train_loss_mateness_history.append(rolling_train_loss_mateness.item() / args['samples_per_epoch'])
        rolling_train_mae_history.append(rolling_train_mae.item() / args['samples_per_epoch'])

        print(f"Train Epoch {epoch + 1}, Loss: {rolling_train_loss.item() / args['samples_per_epoch']:.4f}, MAE: {rolling_train_mae.item() / args['samples_per_epoch']:.4f}")

        # Evaluate the model
        with torch.no_grad():
            model.eval()

            test_loss = 0
            test_loss_win_prob = 0
            test_loss_mateness = 0
            test_mae = 0
            for batch in tqdm(test_loader, desc=f"Test Epoch {epoch + 1}"):
                board_representation, label, win_prob_bins, mateness = batch
                board_representation = board_representation.to(device)
                label = label.to(device)
                win_prob_bins = win_prob_bins.to(device)
                mateness = mateness.to(device)

                win_prob_logits, mateness_logit = model(board_representation)

                win_prob_loss = win_prob_loss_fn(win_prob_logits, win_prob_bins)
                mateness_loss = args['mateness_importance'] * mateness_loss_fn(mateness_logit, mateness)

                loss = win_prob_loss + mateness_loss

                test_loss += loss.detach()
                test_loss_win_prob += win_prob_loss.detach()
                test_loss_mateness += mateness_loss.detach()

                mode = torch.argmax(win_prob_logits, dim=1)
                test_mae += mae(mode.detach() / args['n_bins'] + (1 / (2 * args['n_bins'])), label)

            test_loss_history.append(test_loss.item() / len_test_data)
            test_loss_win_prob_history.append(test_loss_win_prob.item() / len_test_data)
            test_loss_mateness_history.append(test_loss_mateness.item() / len_test_data)
            test_mae_history.append(test_mae.item() / len_test_data)

            print(f"Test Epoch {epoch + 1}, Loss: {test_loss.item() / len_test_data:.4f}, MAE: {test_mae.item() / len_test_data:.4f}")

        # Save the model
        torch.save(model.state_dict(), f"models/{args['name']}_epoch_{epoch + 1}.pth")

        # Save all the losses to a json file
        with open(f"plots/{args['name']}_losses.json", "w") as f:
            json.dump({
                'num_model_parameters': sum(p.numel() for p in model.parameters()),
                'mult-adds': model_summary.total_mult_adds,
                'epochs': args['epochs'],
                'dataset_size': len(train_dataset),
                'samples_per_epoch': args['samples_per_epoch'],
                'rolling_train_loss': rolling_train_loss_history,
                'test_loss': test_loss_history,
                'rolling_train_loss_win_prob': rolling_train_loss_win_prob_history,
                'test_loss_win_prob': test_loss_win_prob_history,
                'rolling_train_loss_mateness': rolling_train_loss_mateness_history,
                'test_loss_mateness': test_loss_mateness_history,
                'rolling_train_mae': rolling_train_mae_history,
                'test_mae': test_mae_history
            }, f)

    # Plot the loss curve
    x = list(range(1, args['epochs'] + 1))
    plt.plot(x, rolling_train_loss_history, label="Rolling Train Loss")
    plt.plot(x, test_loss_history, label="Test Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/{args['name']}_loss_curve.png")
    plt.clf()

    # Plot loss curve split up in win prob and mateness loss
    plt.plot(x, rolling_train_loss_win_prob_history, label="Rolling Train Win Prob Loss")
    plt.plot(x, test_loss_win_prob_history, label="Test Win Prob Loss")
    plt.plot(x, rolling_train_loss_mateness_history, label="Rolling Train Mateness Loss")
    plt.plot(x, test_loss_mateness_history, label="Test Mateness Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/{args['name']}_loss_curve_split.png")
    plt.clf()

    # Plot the MAE curve
    plt.plot(x, rolling_train_mae_history, label="Rolling Train MAE")
    plt.plot(x, test_mae_history, label="Test MAE")
    plt.title("MAE Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/{args['name']}_mae_curve.png")
    plt.clf()

if __name__ == '__main__':
    train_model()
