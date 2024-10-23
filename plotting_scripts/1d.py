from datetime import datetime

import matplotlib.pyplot as plt
import torch
import matplotlib

from FunctionEncoder import TensorboardCallback, FunctionEncoder

import argparse
import os
from tqdm import trange

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_specs import colors, labels, titles

from src.Datasets.DarcyDataset import DarcySrcDataset, DarcyTgtDataset, plot_source_darcy, plot_target_darcy, plot_transformation_darcy
from src.Datasets.HeatDataset import HeatSrcDataset, HeatTgtDataset, plot_source_heat, plot_target_heat, plot_transformation_heat
from src.Datasets.L_shapedDataset import LSrcDataset, LTgtDataset, plot_source_L, plot_target_L, plot_transformation_L
from src.DeepONet import DeepONet
from src.DeepONet_CNN import DeepONet_CNN, DeepONet_2Stage_CNN_branch
from src.MatrixMethodHelpers import compute_A, train_nonlinear_transformation, get_num_parameters, get_num_layers, predict_number_params, get_hidden_layer_size, check_parameters
from src.PODDeepONet import DeepONet_POD
from src.SVDEncoder import SVDEncoder

# import datasets
from src.Datasets.QuadraticSinDataset import QuadraticDataset, SinDataset, plot_source_quadratic, plot_target_sin, plot_transformation_quadratic_sin
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset, plot_source_cubic, plot_target_cubic_derivative, plot_transformation_derivative
from src.Datasets.IntegralDataset import QuadraticIntegralDataset, plot_target_quadratic_integral, plot_transformation_integral
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset, plot_source_mountain_car, plot_target_mountain_car, plot_transformation_mountain_car
from src.Datasets.ElasticPlateDataset import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset,plot_target_boundary, plot_source_boundary_force, plot_transformation_elastic
from src.Datasets.OperatorDataset import CombinedDataset



plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"


def get_dataset(dataset_type:str, test:bool, model_type:str, n_sensors:int, device:str, freeze_example_xs:bool=True, **kwargs):
    # generate datasets
    # freeze_example_xs = model_type in ["deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]  # deeponet has fixed input sensors.
    freeze_xs = model_type in ["deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]
    # NOTE: Most of these datasets are generative, so the data is always unseen, hence no separate test set.
    if dataset_type == "QuadraticSin":
        src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = SinDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Derivative":
        src_dataset = CubicDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device, **kwargs)
        tgt_dataset = CubicDerivativeDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device, **kwargs)
    elif dataset_type == "Integral":
        src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = QuadraticIntegralDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "MountainCar":
        src_dataset = MountainCarPoliciesDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = MountainCarEpisodesDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Elastic":
        src_dataset = ElasticPlateBoudaryForceDataset(freeze_example_xs=freeze_example_xs, test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = ElasticPlateDisplacementDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Darcy":
        src_dataset = DarcySrcDataset(test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = DarcyTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Heat":
        src_dataset = HeatSrcDataset(test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = HeatTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "LShaped":
        src_dataset = LSrcDataset(test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = LTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=(model_type == "matrix"))

    # sample from all of them to freeze the example inputs, which only matters for deeponet.
    src_dataset.sample(device)
    tgt_dataset.sample(device)
    combined_dataset.sample(device)

    return src_dataset, tgt_dataset, combined_dataset

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=10_000)
parser.add_argument("--dataset_type", type=str, default="Derivative")
parser.add_argument("--logdir", type=str, default="logs_experiment")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--approximate_number_paramaters", type=int, default=500_000)
parser.add_argument("--unfreeze_sensors", action="store_true")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
assert 1 <= args.seed <= 10, "Seed must be between 1 and 10." 
assert args.dataset_type in ["Derivative", "Integral", "Darcy"], "Only Derivative, Integral, and Darcy datasets are supported for this script."

# hyper params
epochs = args.epochs
n_basis = args.n_basis
if args.device == "auto": # automatically choose
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device == "cuda" or args.device == "cpu": # use specificed device
    device = args.device
else: # use cuda device at this index
    device = f"cuda:{int(args.device)}"
dataset_type = args.dataset_type
nonlinear_datasets = ["MountainCar", "Elastic", "Darcy", "Heat", "LShaped"]
transformation_type = "nonlinear" if args.dataset_type in nonlinear_datasets else "linear"
n_layers = args.n_layers
freeze_example_xs = not args.unfreeze_sensors

print("Plotting for dataset:", dataset_type)

# first, find the correct dir
logdir = f"{args.logdir}/{dataset_type}"

# next, get all subdirs for both deeponet and b2b (matrix)
deeponet_dir = f"{logdir}/deeponet"
b2b_dir = f"{logdir}/matrix_least_squares"
deeponet_subdirs = [f"{deeponet_dir}/{subdir}" for subdir in os.listdir(deeponet_dir) if os.path.isdir(f"{deeponet_dir}/{subdir}")]
b2b_subdirs = [f"{b2b_dir}/{subdir}" for subdir in os.listdir(b2b_dir) if os.path.isdir(f"{b2b_dir}/{subdir}")]



# next, get the oldest subdir by when it was created
deeponet_subdirs.sort(key=lambda x: datetime.strptime(x.split("/")[-1], "%Y-%m-%d_%H-%M-%S"))
b2b_subdirs.sort(key=lambda x: datetime.strptime(x.split("/")[-1], "%Y-%m-%d_%H-%M-%S"))
deeponet_subdir = deeponet_subdirs[args.seed-1]
b2b_subdir = b2b_subdirs[args.seed-1]

# set the paths in args
args.load_path_matrix = b2b_subdir
args.load_path_deeponet = deeponet_subdir

# generate logdir
logdir = args.load_path_matrix

# seed torch
torch.manual_seed(args.seed)

# generate datasets
src_dataset, tgt_dataset, combined_dataset = get_dataset(dataset_type, test=False, model_type="deeponet", n_sensors=args.n_sensors, device=device, freeze_example_xs=freeze_example_xs)
_, _, testing_combined_dataset = get_dataset(dataset_type, test=True, model_type="deeponet", n_sensors=args.n_sensors, device=device,   freeze_example_xs=freeze_example_xs)

# if using deeponet, we need to copy the input sensors
testing_combined_dataset.src_dataset.example_xs = combined_dataset.src_dataset.example_xs
testing_combined_dataset.example_xs = combined_dataset.example_xs


# FIRST CREATE b2b MODEL
# computes the hidden size that most closely reaches the approximate number of parameters, given a number of layers
hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                    model_type="matrix",
                                    n_basis=n_basis, n_layers=n_layers,
                                    src_input_space=src_dataset.input_size,
                                    src_output_space=src_dataset.output_size,
                                    tgt_input_space=tgt_dataset.input_size,
                                    tgt_output_space=tgt_dataset.output_size,
                                    transformation_type=transformation_type,
                                    n_sensors=combined_dataset.n_examples_per_sample,
                                    dataset_type=dataset_type,)

src_model = FunctionEncoder(input_size=src_dataset.input_size,
                            output_size=src_dataset.output_size,
                            data_type=src_dataset.data_type,
                            n_basis=n_basis,
                            method=args.train_method,
                            model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                            ).to(device)
tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                            output_size=tgt_dataset.output_size,
                            data_type=tgt_dataset.data_type,
                            n_basis=n_basis+1, # note this makes debugging way easier.
                            method=args.train_method,
                            model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                            ).to(device)
b2b_model = {"src": src_model, "tgt": tgt_model}
if transformation_type == "nonlinear":
    transformation_input_size = src_model.n_basis if src_model is not None else src_dataset.output_size[0]
    layers = [torch.nn.Linear(transformation_input_size, hidden_size),torch.nn.ReLU()]
    for layer in range(n_layers - 2):
        layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
    layers += [torch.nn.Linear(hidden_size, tgt_model.n_basis)]
    a_model = torch.nn.Sequential(*layers).to(device)
    b2b_model["A"] = a_model
else:
    b2b_model["A"] = torch.rand(tgt_model.n_basis, src_model.n_basis).to(device)
n_params = get_num_parameters(b2b_model)
predict_n_params = predict_number_params("matrix", combined_dataset.n_examples_per_sample, n_basis, hidden_size, n_layers, src_dataset.input_size, src_dataset.output_size, tgt_dataset.input_size, tgt_dataset.output_size, transformation_type, dataset_type)
assert n_params == predict_n_params, f"Number of parameters is not consistent, expected {predict_n_params}, got {n_params}."


# NEXT CREATE DEEPONET MODEL
hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                    model_type="deeponet",
                                    n_basis=n_basis, n_layers=n_layers,
                                    src_input_space=src_dataset.input_size,
                                    src_output_space=src_dataset.output_size,
                                    tgt_input_space=tgt_dataset.input_size,
                                    tgt_output_space=tgt_dataset.output_size,
                                    transformation_type=transformation_type,
                                    n_sensors=combined_dataset.n_examples_per_sample,
                                    dataset_type=dataset_type,)

deeponet_model = DeepONet(input_size_tgt=tgt_dataset.input_size[0],
                    output_size_tgt=tgt_dataset.output_size[0],
                    input_size_src=src_dataset.input_size[0],
                    output_size_src=src_dataset.output_size[0],
                    n_input_sensors=combined_dataset.n_examples_per_sample,
                    p=n_basis,
                    n_layers=n_layers,
                    hidden_size=hidden_size,
                    ).to(device)
n_params = get_num_parameters(deeponet_model)
predict_n_params = predict_number_params("deeponet", combined_dataset.n_examples_per_sample, n_basis, hidden_size, n_layers, src_dataset.input_size, src_dataset.output_size, tgt_dataset.input_size, tgt_dataset.output_size, transformation_type, dataset_type)
assert n_params == predict_n_params, f"Number of parameters is not consistent, expected {predict_n_params}, got {n_params}."

# load models
b2b_model["src"].load_state_dict(torch.load(f"{args.load_path_matrix}/src_model.pth", weights_only=True, map_location=device))
b2b_model["tgt"].load_state_dict(torch.load(f"{args.load_path_matrix}/tgt_model.pth", weights_only=True, map_location=device))
if transformation_type == "linear":
    b2b_model["A"] = torch.load(f"{args.load_path_matrix}/A.pth", weights_only=True, map_location=device)
else:
    b2b_model["A"].load_state_dict(torch.load(f"{args.load_path_matrix}/A.pth", weights_only=True, map_location=device))

deeponet_model.load_state_dict(torch.load(f"{args.load_path_deeponet}/model.pth", weights_only=True, map_location=device))



##############   Evaluate    ###################
with torch.no_grad():
    
    hardest_example_xs, hardest_example_ys, hardest_xs, hardest_ys, hardest_info = None, None, None, None, None
    b2b_loss = -1000000

    for search_step in trange(1000):
        # plot transformation for all model types
        example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device, plot_only=False)
        
        # get output space predictions
        rep, _ = b2b_model["src"].compute_representation(example_xs, example_ys, method=args.train_method)
        if transformation_type == "linear":
            rep = rep @ b2b_model["A"].T
        else:
            rep = b2b_model["A"](rep)
        b2b_y_hats = b2b_model["tgt"].predict(xs, rep)

        # compute the loss for each function
        losses = ((b2b_y_hats - ys)**2).mean(dim=(1,2))

        # get the hardest example
        max_index = torch.argmax(losses)
        if losses[max_index] > b2b_loss:
            hardest_example_xs = example_xs[max_index:max_index+1]
            hardest_example_ys = example_ys[max_index:max_index+1]
            hardest_xs = xs[max_index:max_index+1]
            hardest_ys = ys[max_index:max_index+1]
            hardest_info = {key: value[max_index:max_index+1] for key, value in info.items()}
            b2b_loss = losses[max_index]

    # use hardest data
    example_xs, example_ys, xs, ys, info = hardest_example_xs, hardest_example_ys, hardest_xs, hardest_ys, hardest_info

    # compute losses
    deeponet_y_hats = deeponet_model.forward(example_xs, example_ys, xs)
    rep, _ = b2b_model["src"].compute_representation(example_xs, example_ys, method=args.train_method)
    if transformation_type == "linear":
        rep = rep @ b2b_model["A"].T
    else:
        rep = b2b_model["A"](rep)
    b2b_y_hats = b2b_model["tgt"].predict(xs, rep)
    b2b_example_y_hats = b2b_model["src"].predict_from_examples(example_xs, example_ys, example_xs, method=args.train_method)

    # PLOT ########################
    size = 5
    b2b_color = colors["matrix_least_squares"]
    b2b_label = labels["matrix_least_squares"]
    deeponet_color = colors["deeponet"]
    deeponet_label = labels["deeponet"]

    # sort example data based on xs
    example_xs, indicies = torch.sort(example_xs, dim=-2)
    example_ys = example_ys.gather(dim=-2, index=indicies)
    b2b_example_y_hats = b2b_example_y_hats.gather(dim=-2, index=indicies)

    # sort data based on xs
    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    b2b_y_hats = b2b_y_hats.gather(dim=-2, index=indicies)
    deeponet_y_hats = deeponet_y_hats.gather(dim=-2, index=indicies)
    print("Saving to logdir:", logdir)

    for row in range(example_xs.shape[0]):
        fig = plt.figure(figsize=(4.25 * size, 1. * size), dpi=300)
        gridspec = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 1, 1])
        axs = gridspec.subplots()

        # plot source space
        ax = axs[0]
        ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth", color="black")
        ax.plot(example_xs[row].cpu(), b2b_example_y_hats[row].cpu(), label=b2b_label, color=b2b_color)
        
        # title depending on data
        if dataset_type == "Derivative":
            title = f"${info['As'][row].item():.2f}x^3 + {info['Bs'][row].item():.2f}x^2 + {info['Cs'][row].item():.2f}x + {info['Ds'][row].item():.2f}$"
        elif dataset_type == "Integral":
            title = f"${info['As'][row].item():.2f}x^2 + {info['Bs'][row].item():.2f}x + {info['Cs'][row].item():.2f}$"
        elif dataset_type == "Darcy":
            title = f"$f(x) \\sim \\mathcal{{G}} \\mathcal{{P}}$"

        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f(x)$")

        # plot source error
        ax = axs[1]
        error = (b2b_example_y_hats[row] - example_ys[row]).abs()
        ax.plot(example_xs[row].cpu(), error.cpu(), label=b2b_label, color=b2b_color)
        ax.set_xlabel("$x$")
        ax.set_ylabel(f"$\\vert \hat{{f}}(x) - f(x) \\vert$")
        ax.set_yscale("log")
        title = f"Absolute Error"
        ax.set_title(title)



        # disable axs[2] so its just white
        ax = axs[2]
        ax.axis("off")


        # plot
        ax = axs[3]
        ax.plot(xs[row].cpu(), ys[row].cpu(), label="Groundtruth", color="black")
        ax.plot(xs[row].cpu(), b2b_y_hats[row].cpu(), label=b2b_label, color=b2b_color)
        ax.plot(xs[row].cpu(), deeponet_y_hats[row].cpu(), label=deeponet_label, color=deeponet_color, ls="--")
        
        if dataset_type == "Derivative":
            title = f"$3*{info['As'][row].item():.2f}x^2 + 2*{info['Bs'][row].item():.2f}x + {info['Cs'][row].item():.2f}$"
        elif dataset_type == "Integral":
            a_string = f"{info['As'][row].item():.2f}"
            b_string = f"{info['Bs'][row].item():.2f}"
            c_string = f"{info['Cs'][row].item():.2f}"
            title = f"$\\frac{{{a_string}}}{{3}}x^3 + \\frac{{{b_string}}}{{2}}x^2  + \\frac{{{c_string}}}{{1}}x$"        
        elif dataset_type == "Darcy":
            title = f"Darcy Solution"
        ax.set_title(title)


        leg = ax.legend(frameon=False)
        # change deeponet linestyle in the legend to be normal, even though its dashed
        for line in leg.get_lines():
            line.set_linestyle('-')

        ax.set_xlabel("$y$")
        ax.set_ylabel("$(\\mathcal{{T}}f)(y)$")


        # plot absolute error
        ax = axs[4]
        error = (b2b_y_hats[row] - ys[row]).abs()
        ax.plot(xs[row].cpu(), error.cpu(), label=b2b_label, color=b2b_color)
        error = (deeponet_y_hats[row] - ys[row]).abs()
        ax.plot(xs[row].cpu(), error.cpu(), label=deeponet_label, color=deeponet_color)
        ax.set_xlabel("$y$")
        ax.set_ylabel(f"$\\vert \hat{{\\mathcal{{T}}}}f(y) - \\mathcal{{T}}f(y) \\vert$")
        ax.set_yscale("log")
        title = f"Absolute Error"
        ax.set_title(title)

        # add line between ax2 and ax3
        left = axs[1].get_position().xmax 
        right = axs[3].get_position().xmin - 0.025
        xpos = (left+right) / 2
        top = axs[1].get_position().ymax + 0.08
        bottom = axs[1].get_position().ymin - 0.08
        line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--", lw=2)
        fig.lines = line1, 


        plt.tight_layout()
        plot_name = f"{logdir}/{dataset_type}_comparison_{row}.pdf"
        plt.savefig(plot_name)


