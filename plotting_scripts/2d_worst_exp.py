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

from src.Datasets.DarcyDataset import DarcySrcDataset, DarcyTgtDataset
from src.Datasets.HeatDataset import HeatSrcDataset, HeatTgtDataset
from src.Datasets.L_shapedDataset import LSrcDataset, LTgtDataset
from src.DeepONet import DeepONet
from src.DeepONet_CNN import DeepONet_CNN, DeepONet_2Stage_CNN_branch
from src.MatrixMethodHelpers import compute_A, train_nonlinear_transformation, get_num_parameters, get_num_layers, predict_number_params, get_hidden_layer_size, check_parameters
from src.PODDeepONet import DeepONet_POD
from src.SVDEncoder import SVDEncoder

# import datasets
from src.Datasets.QuadraticSinDataset import QuadraticDataset, SinDataset
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset
from src.Datasets.IntegralDataset import QuadraticIntegralDataset
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset
from src.Datasets.ElasticPlateDataset import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset
from src.Datasets.OperatorDataset import CombinedDataset

import json
import numpy as np

plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=False)  # Disable LaTeX rendering
plt.rcParams["font.family"] = "serif"  # Use a standard serif font

def get_dataset(dataset_type:str, test:bool, model_type:str, n_sensors:int, device:str, freeze_example_xs:bool=True, **kwargs):
    # generate datasets
    freeze_xs = model_type in ["deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]
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

def plot_predictions(predictions, dataset_type):
    """Plot the predictions and groundtruth data."""
    print("Starting plotting function...")
    
    # Extract data and print shapes for debugging
    b2b_preds = torch.tensor(predictions["b2b_predictions"])
    deeponet_preds = torch.tensor(predictions["deeponet_predictions"])
    groundtruth = torch.tensor(predictions["groundtruth"])
    xs = torch.tensor(predictions["input_data"]["xs"])
    
    print(f"Shapes:")
    print(f"b2b_preds: {b2b_preds.shape}")
    print(f"deeponet_preds: {deeponet_preds.shape}")
    print(f"groundtruth: {groundtruth.shape}")
    print(f"xs: {xs.shape}")
    
    # Let's also print some sample values to check the data
    print("\nSample values at first few points:")
    print(f"xs[0,:5]: {xs[0,:5]}")
    print(f"groundtruth[0,:5]: {groundtruth[0,:5]}")
    print(f"b2b_preds[0,:5]: {b2b_preds[0,:5]}")
    print(f"deeponet_preds[0,:5]: {deeponet_preds[0,:5]}")
    
    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    if dataset_type == "LShaped":
        print("Plotting LShaped dataset...")
        # Basic quiver plot without any modifications
        ax[0].quiver(xs[0,:,0].cpu(), xs[0,:,1].cpu(), 
                    groundtruth[0,:,0].cpu(), groundtruth[0,:,1].cpu())
        ax[1].quiver(xs[0,:,0].cpu(), xs[0,:,1].cpu(), 
                    b2b_preds[0,:,0].cpu(), b2b_preds[0,:,1].cpu())
        ax[2].quiver(xs[0,:,0].cpu(), xs[0,:,1].cpu(), 
                    deeponet_preds[0,:,0].cpu(), deeponet_preds[0,:,1].cpu())
    
    # Set titles and labels
    ax[0].set_title('Ground Truth')
    ax[1].set_title('B2B Prediction')
    ax[2].set_title('DeepONet Prediction')
    
    for a in ax:
        a.set_xlabel('x')
        a.set_ylabel('y')
        a.set_aspect('equal')
        a.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "prediction_plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"predictions_{dataset_type}.png")
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot instead of closing it
    plt.show()
    print("Plotting completed!")

def main():
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
    
    # Print selected directories
    print(f"Selected DeepONet directory: {deeponet_subdir}")
    print(f"Selected B2B directory: {b2b_subdir}")
    
    # Verify file existence before loading
    src_model_path = os.path.join(b2b_subdir, "src_model.pth")
    tgt_model_path = os.path.join(b2b_subdir, "tgt_model.pth")
    a_model_path = os.path.join(b2b_subdir, "A.pth")
    
    if not os.path.exists(src_model_path):
        raise FileNotFoundError(f"Source model not found at: {src_model_path}")
    if not os.path.exists(tgt_model_path):
        raise FileNotFoundError(f"Target model not found at: {tgt_model_path}")
    if not os.path.exists(a_model_path):
        raise FileNotFoundError(f"A model not found at: {a_model_path}")

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
    with torch.no_grad():
        # Add offset variable here
        minus_worst = -9  # Offset from worst case sample
        
        # First find the worst case
        hardest_example_xs, hardest_example_ys, hardest_xs, hardest_ys, hardest_info = None, None, None, None, None
        b2b_loss = -1000000
        hardest_index = -1

        # Find worst case index
        for search_step in trange(1000):
            example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device, plot_only=False)
            
            if type(combined_dataset.src_dataset) == HeatSrcDataset:
                rep = example_ys[:, 0, :]
            else:
                rep, _ = b2b_model["src"].compute_representation(example_xs, example_ys, method=args.train_method)

            if transformation_type == "linear":
                rep = rep @ b2b_model["A"].T
            else:
                rep = b2b_model["A"](rep)
            b2b_y_hats = b2b_model["tgt"].predict(xs, rep)
            
            losses = ((b2b_y_hats - ys)**2).mean(dim=(1,2))
            max_index = torch.argmax(losses)
            
            if losses[max_index] > b2b_loss:
                b2b_loss = losses[max_index]
                hardest_index = info['function_indicies'][max_index].item()

        # Calculate offset index
        offset_index = hardest_index + minus_worst
        print(f"\nWorst case index: {hardest_index}")
        print(f"Using offset sample index: {offset_index}")

        # Get the offset sample
        found_offset = False
        while not found_offset:
            example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device, plot_only=False)
            if 'function_indicies' in info and offset_index in info['function_indicies']:
                idx = (info['function_indicies'] == offset_index).nonzero(as_tuple=True)[0][0]
                example_xs = example_xs[idx:idx+1]
                example_ys = example_ys[idx:idx+1]
                xs = xs[idx:idx+1]
                ys = ys[idx:idx+1]
                info = {key: value[idx:idx+1] for key, value in info.items()}
                found_offset = True

        # Now use these offset samples for predictions
        if args.dataset_type == "Heat":
            function_indicies = info["function_indicies"]
            all_xs = testing_combined_dataset.tgt_dataset.xs[function_indicies]
            all_ys = testing_combined_dataset.tgt_dataset.ys[function_indicies]
            xs = all_xs[:, 49::99, :]
            ys = all_ys[:, 49::99, :]
            grid = example_xs
            grid_outs = example_ys
        else:
            grid = example_xs
            grid_outs = example_ys

        # Compute predictions using the offset sample
        if dataset_type != "Heat":
            b2b_example_y_hats = b2b_model["src"].predict_from_examples(example_xs, example_ys, grid, method=args.train_method)
        else:
            b2b_example_y_hats = None

        if type(combined_dataset.src_dataset) == HeatSrcDataset:
            rep = example_ys[:, 0, :]
        else:
            rep, _ = b2b_model["src"].compute_representation(example_xs, example_ys, method=args.train_method)

        if transformation_type == "linear":
            rep = rep @ b2b_model["A"].T
        else:
            rep = b2b_model["A"](rep)
            
        b2b_y_hats = b2b_model["tgt"].predict(xs, rep)
        deeponet_y_hats = deeponet_model.forward(example_xs, example_ys, xs)

        # Save predictions
        predictions = {
            "b2b_predictions": b2b_y_hats.cpu().tolist(),
            "deeponet_predictions": deeponet_y_hats.cpu().tolist(),
            "groundtruth": ys.cpu().tolist(),
            "input_data": {
                "example_xs": example_xs.cpu().tolist(),
                "example_ys": example_ys.cpu().tolist(),
                "xs": xs.cpu().tolist(),
                "ys": ys.cpu().tolist(),
            },
            "additional_info": {k: v.cpu().tolist() for k, v in info.items()},
            "indices_info": {
                "worst_case_index": hardest_index,
                "offset_index": offset_index,
                "offset_amount": minus_worst
            }
        }

        # Define a directory to save predictions
        predictions_dir = "predictions_output_101"
        os.makedirs(predictions_dir, exist_ok=True)

        # Save to a JSON file
        prediction_file = os.path.join(predictions_dir, f"predictions_seed_101.json")
        try:
            with open(prediction_file, "w") as f:
                json.dump(predictions, f, indent=4)
            print(f"Predictions saved to {prediction_file}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")

        # After saving predictions, add:
        try:
            print("\nStarting plotting process...")
            plot_predictions(predictions, args.dataset_type)
            print("Plot saved successfully!")
        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()