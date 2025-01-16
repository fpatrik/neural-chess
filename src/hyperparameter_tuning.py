import json
import matplotlib.pyplot as plt
import numpy as np

from src.train import DEFAULT_ARGS, train_model

if __name__ == '__main__':

    # Scaling laws for number of parameters
    very_small_args = {
        **DEFAULT_ARGS,
        'name': 'parameters_very_small',
        'samples_per_epoch': 2_000_000,
        'restrict_dataset_size': 100_000_000,
        'epochs': 50,
        'model_args': {
            'input_dim': 56,
            'model_dim': 4,
            'bin_dim': 128,
            'n_heads': 1,
            'num_layers': 6,
            'dim_feedforward': 8,
            'dropout': 0
        },
    }

    train_model(very_small_args)

    small_args = {
        **DEFAULT_ARGS,
        'name': 'parameters_small',
        'samples_per_epoch': 2_000_000,
        'restrict_dataset_size': 100_000_000,
        'epochs': 100,
        'model_args': {
            'input_dim': 56,
            'model_dim': 8,
            'bin_dim': 128,
            'n_heads': 1,
            'num_layers': 6,
            'dim_feedforward': 16,
            'dropout': 0
        },
    }

    train_model(small_args)

    medium_args = {
        **DEFAULT_ARGS,
        'name': 'parameters_medium',
        'samples_per_epoch': 2_000_000,
        'restrict_dataset_size': 100_000_000,
        'epochs': 200,
        'model_args': {
            'input_dim': 56,
            'model_dim': 16,
            'bin_dim': 128,
            'n_heads': 2,
            'num_layers': 6,
            'dim_feedforward': 32,
            'dropout': 0
        },
    }

    train_model(medium_args)

    large_args = {
        **DEFAULT_ARGS,
        'name': 'parameters_large',
        'samples_per_epoch': 2_000_000,
        'restrict_dataset_size': 100_000_000,
        'epochs': 400,
        'model_args': {
            'input_dim': 56,
            'model_dim': 32,
            'bin_dim': 128,
            'n_heads': 2,
            'num_layers': 6,
            'dim_feedforward': 64,
            'dropout': 0
        },
    }

    train_model(large_args)

    # Read the results
    n_parameters = []
    test_losses = []
    final_test_losses = []
    computes = []
    samples = []
    for args in [very_small_args, small_args, medium_args, large_args]:
        with open(f"plots/{args['name']}_losses.json") as f:
            result = json.load(f)
            n_parameters.append(float(result['num_model_parameters']))
            test_losses.append(result['test_loss'])
            final_test_losses.append(result['test_loss'][-1])

            computes.append([result['mult-adds'] * (1 + e) for e in range(result['epochs'])])
            samples.append([result['samples_per_epoch'] * (1 + e) for e in range(result['epochs'])])

    # Log-log plot of test loss vs number of parameters
    plt.loglog(n_parameters, final_test_losses, marker = 'o')
    plt.xlabel("Number of parameters")
    plt.ylabel("Test loss")
    plt.title("Parameter Scaling Laws")
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(f"plots/scaling_law_parameters.png")
    plt.clf()

    # Log-log plot of test loss vs samples
    for i, sample in enumerate(samples):
        plt.loglog(sample, test_losses[i], label = f"{int(np.ceil(n_parameters[i] / 1e3))}k parameters")
    plt.xlabel("Samples")
    plt.ylabel("Test loss")
    plt.title("Sample Efficiency")
    plt.legend()
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(f"plots/sample_efficiency.png")
    plt.clf()


    # Log-log plot of test loss vs compute
    max_compute = max([max(compute) for compute in computes])
    for i, compute in enumerate(computes):
        plt.loglog(np.array(compute) / max_compute, test_losses[i], label = f"{int(np.ceil(n_parameters[i] / 1e3))}k parameters")
    plt.xlabel("Compute")
    plt.ylabel("Test loss")
    plt.title("Compute Scaling Laws")
    plt.legend()
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(f"plots/scaling_law_compute.png")
    plt.clf()
