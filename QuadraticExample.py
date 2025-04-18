from datetime import datetime

import matplotlib.pyplot as plt
import torch

# Import only the cubic dataset, not the derivative dataset
from src.Datasets.DerivativeDataset import CubicDataset, plot_source_cubic
from FunctionEncoder import FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=15)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=100000)  # 700000 / 20 = 35000 epochs
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--n_functions_per_batch", type=int, default=50)  # 20 functions per batch
parser.add_argument("--n_output_points", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=1e-3)  # Initial learning rate of 10^-3
parser.add_argument("--lr_decay_step", type=int, default=30000)  # Step at which to decrease learning rate
parser.add_argument("--lr_decay_factor", type=float, default=0.1)  # Factor to decrease learning rate by
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
n_sensors = args.n_sensors
n_functions_per_batch = args.n_functions_per_batch
n_output_points = args.n_output_points
learning_rate = args.learning_rate
lr_decay_step = args.lr_decay_step
lr_decay_factor = args.lr_decay_factor
residuals = args.residuals
if load_path is None:
    logdir = f"logs/cubic_source_only/{train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "MLP" if not args.parallel else "ParallelMLP"

# seed torch
torch.manual_seed(seed)

# create a dataset for the cubic function (source domain only)
a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
d_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = CubicDataset(a_range=a_range, b_range=b_range, c_range=c_range, d_range=d_range, 
                     input_range=input_range, 
                     n_examples_per_sample=n_sensors,
                     n_functions_per_sample=n_functions_per_batch,
                     n_points_per_sample=n_output_points)

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, device=device, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample(device)
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    if train_method == "inner_product":
        y_hats_ip = model.predict_from_examples(example_xs, example_ys, query_xs, method="inner_product")
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")
    
    # Sort for better plotting
    query_xs, indices = torch.sort(query_xs, dim=-2)
    query_ys = query_ys.gather(dim=-2, index=indices)
    y_hats_ls = y_hats_ls.gather(dim=-2, index=indices)
    if train_method == "inner_product":
        y_hats_ip = y_hats_ip.gather(dim=-2, index=indices)

    # Plot cubic functions and their predictions
    plot_source_cubic(query_xs, query_ys, y_hats_ls, info, logdir)

    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    query_xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
    
    # Use the correct method for getting basis functions from model
    basis = model.model.forward(query_xs)
    
    for i in range(n_basis):
        ax.plot(query_xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    if residuals:
        avg_function = model.average_function.forward(query_xs)
        ax.plot(query_xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    plt.title("Basis Functions for Cubic Functions")
    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")

