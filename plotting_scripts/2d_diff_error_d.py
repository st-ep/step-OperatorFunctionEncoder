from datetime import datetime

import matplotlib.pyplot as plt
import torch
import matplotlib
import copy

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
from src.Datasets.ElasticDataError import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset,plot_target_boundary, plot_source_boundary_force, plot_transformation_elastic
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
parser.add_argument("--dataset_type", type=str, default="Elastic")
parser.add_argument("--logdir", type=str, default="logs_experiment")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--approximate_number_paramaters", type=int, default=500_000)
parser.add_argument("--unfreeze_sensors", action="store_true")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
assert 1 <= args.seed <= 10, "Seed must be between 1 and 10." 
assert args.dataset_type in ["Elastic", "Heat", "LShaped"], "Only Elastic, Heat, and LShaped are supported."

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

if dataset_type != "Heat":
    src_model = FunctionEncoder(input_size=src_dataset.input_size,
                                    output_size=src_dataset.output_size,
                                    data_type=src_dataset.data_type,
                                    n_basis=n_basis,
                                    method=args.train_method,
                                    model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                    ).to(device)
else:
    src_model = None # heat dataset has no source space, just temperature and alpha
tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                            output_size=tgt_dataset.output_size,
                            data_type=tgt_dataset.data_type,
                            n_basis=n_basis+1, # note this makes debugging way easier.
                            method=args.train_method,
                            model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                            ).to(device)
b2b_model = {"src": src_model, "tgt": tgt_model}

# optionally add neural network to transform between spaces for nonlinear operator
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
if b2b_model["src"] is not None:
    b2b_model["src"].load_state_dict(torch.load(f"{args.load_path_matrix}/src_model.pth", weights_only=True, map_location=device))
b2b_model["tgt"].load_state_dict(torch.load(f"{args.load_path_matrix}/tgt_model.pth", weights_only=True, map_location=device))
if transformation_type == "linear":
    b2b_model["A"] = torch.load(f"{args.load_path_matrix}/A.pth", weights_only=True, map_location=device)
else:
    b2b_model["A"].load_state_dict(torch.load(f"{args.load_path_matrix}/A.pth", weights_only=True, map_location=device))

deeponet_model.load_state_dict(torch.load(f"{args.load_path_deeponet}/model.pth", weights_only=True, map_location=device))



##############   Evaluate    ###################
# Define denormalize function before the search loop
def denormalize(y):
    mean, std = (2.8366714104777202e-05, 4.263603113940917e-05)
    return y * std + mean

with torch.no_grad():
    hardest_example_xs, hardest_example_ys, hardest_xs, hardest_ys, hardest_info = None, None, None, None, None
    max_relative_loss = -1000000
    hardest_index = -1
    found_valid_example = False

    for search_step in trange(1000):
        example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device, plot_only=False)
        
        # Only consider samples that have at least one index in the range 100-150
        valid_indices_mask = (info['function_indicies'] >= 100) & (info['function_indicies'] <= 150)
        if not valid_indices_mask.any():
            continue

        found_valid_example = True
        
        # Get DeepONet predictions
        deeponet_y_hats = deeponet_model.forward(example_xs, example_ys, xs)
        
        # Denormalize predictions and ground truth
        ys_denorm = denormalize(ys)
        deeponet_y_hats_denorm = denormalize(deeponet_y_hats)
        
        # Compute relative losses for each function
        max_displacements = ys_denorm.abs().max(dim=1)[0].max(dim=1)[0]  # Maximum displacement for each sample
        relative_losses = torch.mean(torch.abs(deeponet_y_hats_denorm - ys_denorm), dim=(1,2)) / max_displacements

        # Only consider losses for indices in our target range
        relative_losses[~valid_indices_mask] = -float('inf')  # Set losses for invalid indices to -inf

        # get the hardest example based on relative loss
        max_index = torch.argmax(relative_losses)
        if relative_losses[max_index] > max_relative_loss:
            hardest_example_xs = example_xs[max_index:max_index+1]
            hardest_example_ys = example_ys[max_index:max_index+1]
            hardest_xs = xs[max_index:max_index+1]
            hardest_ys = ys[max_index:max_index+1]
            hardest_info = {key: value[max_index:max_index+1] for key, value in info.items()}
            max_relative_loss = relative_losses[max_index]
            hardest_index = search_step

    if not found_valid_example:
        raise RuntimeError("No valid examples found in the last 50 samples. Please check the dataset indices.")

    print(f"\nHardest example function index (for DeepONet): {hardest_info['function_indicies'].item()}")
    print(f"Maximum relative loss value (DeepONet): {(max_relative_loss * 100):.4f}%")

    # Get ground truth displacements
    hardest_index = hardest_info['function_indicies'].item()
    info_hardest = {'function_indicies': torch.tensor([hardest_index], device=device)}
    info_offset1 = {'function_indicies': torch.tensor([hardest_index - 50], device=device)}
    info_offset2 = {'function_indicies': torch.tensor([hardest_index - 100], device=device)}

    # Sample inputs and get ground truth outputs for all cases
    xs_hardest = testing_combined_dataset.tgt_dataset.sample_inputs(info_hardest, None)
    xs_offset1 = testing_combined_dataset.tgt_dataset.sample_inputs(info_offset1, None)
    xs_offset2 = testing_combined_dataset.tgt_dataset.sample_inputs(info_offset2, None)
    
    ys_hardest = testing_combined_dataset.tgt_dataset.compute_outputs(info_hardest, xs_hardest)
    ys_offset1 = testing_combined_dataset.tgt_dataset.compute_outputs(info_offset1, xs_offset1)
    ys_offset2 = testing_combined_dataset.tgt_dataset.compute_outputs(info_offset2, xs_offset2)

    # Get corresponding source inputs and outputs
    example_xs_hardest = testing_combined_dataset.src_dataset.sample_inputs(info_hardest, None)
    example_xs_offset1 = testing_combined_dataset.src_dataset.sample_inputs(info_offset1, None)
    example_xs_offset2 = testing_combined_dataset.src_dataset.sample_inputs(info_offset2, None)
    
    example_ys_hardest = testing_combined_dataset.src_dataset.compute_outputs(info_hardest, example_xs_hardest)
    example_ys_offset1 = testing_combined_dataset.src_dataset.compute_outputs(info_offset1, example_xs_offset1)
    example_ys_offset2 = testing_combined_dataset.src_dataset.compute_outputs(info_offset2, example_xs_offset2)

    rep_hardest, _ = b2b_model["src"].compute_representation(example_xs_hardest, example_ys_hardest, method=args.train_method)
    rep_offset1, _ = b2b_model["src"].compute_representation(example_xs_offset1, example_ys_offset1, method=args.train_method)
    rep_offset2, _ = b2b_model["src"].compute_representation(example_xs_offset2, example_ys_offset2, method=args.train_method)

    rep_hardest = b2b_model["A"](rep_hardest)
    rep_offset1 = b2b_model["A"](rep_offset1)
    rep_offset2 = b2b_model["A"](rep_offset2)

    b2b_y_hardest = b2b_model["tgt"].predict(xs_hardest, rep_hardest)
    b2b_y_offset1 = b2b_model["tgt"].predict(xs_offset1, rep_offset1)
    b2b_y_offset2 = b2b_model["tgt"].predict(xs_offset2, rep_offset2)

    # Get DeepONet predictions
    deeponet_y_hardest = deeponet_model.forward(example_xs_hardest, example_ys_hardest, xs_hardest)
    deeponet_y_offset1 = deeponet_model.forward(example_xs_offset1, example_ys_offset1, xs_offset1)
    deeponet_y_offset2 = deeponet_model.forward(example_xs_offset2, example_ys_offset2, xs_offset2)

    # Add after getting the ground truth outputs
    def denormalize(y):
        mean, std = (2.8366714104777202e-05, 4.263603113940917e-05)
        return y * std + mean

    # Denormalize before checking linearity
    ys_hardest_denorm = denormalize(ys_hardest)
    ys_offset1_denorm = denormalize(ys_offset1)
    ys_offset2_denorm = denormalize(ys_offset2)

    # Compare ground truth linearity with denormalized values
    gt_sum = ys_offset1_denorm + ys_offset2_denorm
    gt_diff = torch.abs(ys_hardest_denorm - gt_sum)
    print(f"\nGround Truth Linearity Check (Denormalized):")
    print(f"Max difference: {torch.max(gt_diff):.6e}")
    print(f"Mean difference: {torch.mean(gt_diff):.6e}")

    # Compare matrix model linearity with denormalized values
    b2b_y_hardest_denorm = denormalize(b2b_y_hardest)
    b2b_y_offset1_denorm = denormalize(b2b_y_offset1)
    b2b_y_offset2_denorm = denormalize(b2b_y_offset2)
    b2b_sum = 10*b2b_y_offset1_denorm + 15*b2b_y_offset2_denorm
    b2b_diff = torch.abs(b2b_y_hardest_denorm - b2b_sum)
    print(f"\nMatrix Model Linearity Check (Denormalized):")
    print(f"Max difference: {torch.max(b2b_diff):.6e}")
    print(f"Mean difference: {torch.mean(b2b_diff):.6e}")

    # Compare DeepONet linearity with denormalized values
    deeponet_y_hardest_denorm = denormalize(deeponet_y_hardest)
    deeponet_y_offset1_denorm = denormalize(deeponet_y_offset1)
    deeponet_y_offset2_denorm = denormalize(deeponet_y_offset2)
    deeponet_sum = 10*deeponet_y_offset1_denorm + 15*deeponet_y_offset2_denorm
    deeponet_diff = torch.abs(deeponet_y_hardest_denorm - deeponet_sum)
    print(f"\nDeepONet Linearity Check (Denormalized):")
    print(f"Max difference: {torch.max(deeponet_diff):.6e}")
    print(f"Mean difference: {torch.mean(deeponet_diff):.6e}")

    # use hardest data
    example_xs, example_ys, xs, ys, info = hardest_example_xs, hardest_example_ys, hardest_xs, hardest_ys, hardest_info

    # Add offset variable
    _diff1 = 100
    _diff2 = 50
    # Get new data for offset index
    new_index1 = hardest_info['function_indicies'].item() - _diff1
    new_index2 = hardest_info['function_indicies'].item() - _diff2
    info1 = {'function_indicies': torch.tensor([new_index1], device=device)}
    info2 = {'function_indicies': torch.tensor([new_index2], device=device)}
    example_xs1 = testing_combined_dataset.src_dataset.sample_inputs(info1, None)
    example_ys1 = testing_combined_dataset.src_dataset.compute_outputs(info1, example_xs1)
    example_xs2 = testing_combined_dataset.src_dataset.sample_inputs(info2, None)
    example_ys2 = testing_combined_dataset.src_dataset.compute_outputs(info2, example_xs2)
    xs1 = testing_combined_dataset.tgt_dataset.sample_inputs(info1, None)
    xs2 = testing_combined_dataset.tgt_dataset.sample_inputs(info2, None)
    ys1 = testing_combined_dataset.tgt_dataset.compute_outputs(info1, xs1)
    ys2 = testing_combined_dataset.tgt_dataset.compute_outputs(info2, xs2)

    # Use xs1, for plotting since we're only showing one example
    xs = xs1
    example_xs = example_xs1
    example_ys = example_ys1

    # fetch the correct plotting functions
    if args.dataset_type == "Elastic":
        plot_source = plot_source_boundary_force
        plot_target = plot_target_boundary
        plot_transformation = plot_transformation_elastic
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    grid = example_xs1
    grid_outs = example_ys1

    # first compute example y_hats 
    b2b_example_y_hats = None
    deeponet_example_y_hats = None

    dict_b2b = copy.deepcopy(info1)
    dict_b2b["model_type"] = "matrix_least_squares"
    dict_deeponet = copy.deepcopy(info1)
    dict_deeponet["model_type"] = "deeponet"
    
    plot_transformation(grid, grid_outs, b2b_example_y_hats, xs, b2b_y_hardest_denorm, b2b_sum, dict_b2b, args.load_path_matrix)
    plot_transformation(grid, grid_outs, deeponet_example_y_hats, xs, deeponet_y_hardest_denorm, deeponet_sum, dict_deeponet, args.load_path_deeponet)

    # Calculate and print average relative losses for the hardest sample
    max_displacement = ys_hardest_denorm.abs().max()
    b2b_relative_loss = torch.mean(torch.abs(b2b_y_hardest_denorm - ys_hardest_denorm)) / max_displacement
    deeponet_relative_loss = torch.mean(torch.abs(deeponet_y_hardest_denorm - ys_hardest_denorm)) / max_displacement
    
    print(f"\nRelative Losses (MAE) for hardest sample:")
    print(f"B2B Relative Loss: {(b2b_relative_loss * 100):.4f}%")
    print(f"DeepONet Relative Loss: {(deeponet_relative_loss * 100):.4f}%")
    print(f"Relative Difference: {(abs(b2b_relative_loss - deeponet_relative_loss) * 100):.4f}%")
    print(f"Ratio: {deeponet_relative_loss / b2b_relative_loss:.4f}")

