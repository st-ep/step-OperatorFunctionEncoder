from datetime import datetime
# here
import matplotlib.pyplot as plt
import torch
import json

from FunctionEncoder import TensorboardCallback, FunctionEncoder

import argparse
import os
from tqdm import trange

from src.Datasets.DarcyDataset import DarcySrcDataset, DarcyTgtDataset, plot_source_darcy, plot_target_darcy, plot_transformation_darcy
from src.Datasets.HeatDataset import HeatSrcDataset, HeatTgtDataset, plot_source_heat, plot_target_heat, plot_transformation_heat
from src.Datasets.L_shapedDataset import LSrcDataset, LTgtDataset, plot_source_L, plot_target_L, plot_transformation_L
from src.DeepONet import DeepONet
from src.DeepONet_CNN import DeepONet_CNN, DeepONet_2Stage_CNN_branch
from src.MatrixMethodHelpers import compute_A, train_nonlinear_transformation, get_num_parameters, get_num_layers, predict_number_params, get_hidden_layer_size, check_parameters
from src.PODDeepONet import DeepONet_POD
from src.SVDEncoder import SVDEncoder
from src.DeepOSet import DeepOSet

# import datasets
from src.Datasets.QuadraticSinDataset import QuadraticDataset, SinDataset, plot_source_quadratic, plot_target_sin, plot_transformation_quadratic_sin
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset, plot_source_cubic, plot_target_cubic_derivative, plot_transformation_derivative
from src.Datasets.IntegralDataset import QuadraticIntegralDataset, plot_target_quadratic_integral, plot_transformation_integral
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset, plot_source_mountain_car, plot_target_mountain_car, plot_transformation_mountain_car
from src.Datasets.ElasticPlateDataset import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset,plot_target_boundary, plot_source_boundary_force, plot_transformation_elastic
from src.Datasets.BurgerDataset import BurgerInputDataset, BurgerOutputDataset, plot_source_burger, plot_target_burger, plot_transformation_burger
from src.Datasets.OperatorDataset import CombinedDataset


def get_dataset(dataset_type:str, test:bool, model_type:str, n_sensors:int, device:str, freeze_example_xs_train:bool=True, test_variable_sensors:bool=False, no_copy_sensors_to_test:bool=False, **kwargs):
    # generate datasets
    # freeze_example_xs = model_type in ["deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]  # deeponet has fixed input sensors.
    # DeepOSet can handle varying sensors, but freeze by default like DeepONet for comparison unless overridden

    # Determine freezing for the specific dataset being created (train or test)
    if test:
        # For the TEST dataset:
        # Freeze source sensors ONLY IF training sensors were frozen AND we ARE copying sensors OR NOT testing variable sensors
        current_freeze_example_xs = freeze_example_xs_train and not (test_variable_sensors and no_copy_sensors_to_test)
    else:
        # For the TRAIN dataset:
        current_freeze_example_xs = freeze_example_xs_train

    # Target query freezing depends only on model type (POD/2Stage freeze, others don't by default)
    freeze_xs = model_type in ["deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]

    # NOTE: Most of these datasets are generative, so the data is always unseen, hence no separate test set.
    if dataset_type == "QuadraticSin":
        src_dataset = QuadraticDataset(freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = SinDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Derivative":
        src_dataset = CubicDataset(freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device, **kwargs)
        tgt_dataset = CubicDerivativeDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device, **kwargs)
    elif dataset_type == "Integral":
        src_dataset = QuadraticDataset(freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = QuadraticIntegralDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "MountainCar":
        src_dataset = MountainCarPoliciesDataset(freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = MountainCarEpisodesDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Elastic":
        src_dataset = ElasticPlateBoudaryForceDataset(freeze_example_xs=current_freeze_example_xs, test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = ElasticPlateDisplacementDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Darcy":
        src_dataset = DarcySrcDataset(test=test, freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = DarcyTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Heat":
        src_dataset = HeatSrcDataset(test=test, freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = HeatTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "LShaped":
        src_dataset = LSrcDataset(test=test, freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = LTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Burger":
        src_dataset = BurgerInputDataset(test=test, freeze_example_xs=current_freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = BurgerOutputDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=(model_type == "matrix"))

    return src_dataset, tgt_dataset, combined_dataset

# test any model on a dataset
def test(model,
         testing_combined_dataset:CombinedDataset, # Renamed parameter for clarity
         callback:TensorboardCallback,
         transformation_type:str,
         train_method:str,
         model_type:str ):
    # set combined dataset to give us more testing data.
    if model_type == "matrix":
        # Ensure the testing dataset instance is used here too
        testing_combined_dataset.calibration_only = False

    with torch.no_grad():
        num_trials = 10
        loss = 0
        for test_iter in range(num_trials): # Renamed loop variable
            # Get data from the TESTING dataset
            src_xs, src_ys, tgt_xs, tgt_ys, info = testing_combined_dataset.sample(device)

            # Compute y_hats for a given model type
            if model_type == "matrix":

                # note the heat dataset has no source space, so the representation is simply alpha, temperature
                # Use the testing dataset instance here
                if type(testing_combined_dataset.src_dataset) == HeatSrcDataset:
                    src_Cs = src_ys[:, 0, :]
                else: # otherwise we compute the representation from data.
                    src_Cs, _ = model["src"].compute_representation(src_xs, src_ys, method=train_method) # Use args.train_method?
                if transformation_type == "linear":
                    tgt_Cs_hat = src_Cs @ model["A"].T
                else:
                    tgt_Cs_hat = model["A"](src_Cs)
                tgt_y_hats = model["tgt"].predict(tgt_xs, tgt_Cs_hat)
            elif model_type == "SVD" or model_type == "Eigen":
                tgt_y_hats = model.predict_from_examples(src_xs, src_ys, tgt_xs, method=train_method, representation_dataset="source", prediction_dataset="target")
            elif model_type == "deeponet_2stage":
                tgt_Cs_hat = (model["T"] @ model["A"](src_ys.reshape(src_ys.shape[0], -1)).T).T
                tgt_y_hats = model["tgt"].predict(tgt_xs, tgt_Cs_hat)
            elif model_type == "deeponet_2stage_cnn":
                tgt_Cs_hat = (model["T"] @ model["A"](src_ys).T).T
                tgt_y_hats = model["tgt"].predict(tgt_xs, tgt_Cs_hat)
            else: # deeponet*, deeposet
                tgt_y_hats = model.forward(src_xs, src_ys, tgt_xs)

            # Compute loss
            loss += torch.nn.MSELoss()(tgt_y_hats, tgt_ys)
        loss = loss / num_trials

    # log under a new tag
    callback.tensorboard.add_scalar("test/mse", loss.item(), callback.total_epochs)

    # Set combined dataset back to training mode for matrix method
    if model_type == "matrix":
        # Use the testing dataset instance
        testing_combined_dataset.calibration_only = True



# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=10_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_type", type=str, default="matrix")
parser.add_argument("--dataset_type", type=str, default="Derivative")
parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--approximate_number_paramaters", type=int, default=300_000)
# --- DeepOSet Specific Args ---
parser.add_argument("--phi_hidden_size", type=int, default=256, help="Hidden size for DeepOSet phi network")
parser.add_argument("--rho_hidden_size", type=int, default=256, help="Hidden size for DeepOSet rho network")
parser.add_argument("--trunk_hidden_size", type=int, default=256, help="Hidden size for DeepOSet trunk network")
parser.add_argument("--pos_encoding_type", type=str, default='skip', choices=['mlp', 'sinusoidal', 'skip'], help="Type of positional encoding for DeepOSet ('mlp', 'sinusoidal', or 'skip' to use raw positions)")
parser.add_argument("--pos_encoding_dim", type=int, default=64, help="Dimension for MLP positional encoding output (concatenate) or sinusoidal features (film)") # default 64
parser.add_argument("--pos_encoding_max_freq", type=float, default=100.0, help="Maximum frequency/scale for sinusoidal positional encoding in DeepOSet") # Make this 10.0
parser.add_argument("--encoding_strategy", type=str, default="concatenate",
                   choices=['concatenate', 'film'], # Removed 'function_encoder'
                   help="Encoding strategy for DeepOSet branch ('concatenate' or 'film')")
parser.add_argument("--film_modulation_dim", type=int, default=None, help="Modulation dimension for FiLM strategy (defaults to phi_hidden_size if None)")
parser.add_argument("--phi_output_size", type=int, default=32, # default 32
                   help="Output dimension of the phi network in DeepOSet before aggregation")
parser.add_argument("--aggregation_type", type=str, default="mean",
                    choices=["mean", "attention"],
                    help="Sensor aggregation method for DeepOSet branch ('mean' or 'attention')")
parser.add_argument("--attention_n_tokens", type=int, default=8,
                    help="Number of learnable query tokens if aggregation_type='attention'")
# --- End DeepOSet Specific Args ---
parser.add_argument("--unfreeze_sensors", action="store_true")
parser.add_argument("--use_lr_schedule", action="store_true", help="Enable learning rate scheduling for applicable models (e.g., DeepONet)")
parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[50000, 100000, 150000, 200000, 250000], help="List of steps (iterations) for LR decay milestones.")
parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2], help="List of multiplicative factors (gammas) for LR decay at each step.")
parser.add_argument("--test_variable_sensors", action="store_true", help="If set, train with fixed source sensors (unless --unfreeze_sensors) and variable target queries, but test with variable source sensors and variable target queries.")
parser.add_argument("--no_copy_sensors_to_test", action="store_true",
                   help="If set, sensor locations will not be copied from training to testing, allowing fully independent test sensor locations.")

args = parser.parse_args()
assert args.model_type in ["SVD", "Eigen", "matrix", "deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn", "deeposet"]
assert args.dataset_type in ["QuadraticSin", "Derivative", "Integral",  "Elastic", "Darcy", "Heat", "LShaped", "Burger"]

# cancel bad combinations
check_parameters(args)

# Validate LR schedule arguments if schedule is used
if args.use_lr_schedule:
    if not args.lr_schedule_steps or not args.lr_schedule_gammas:
        parser.error("--lr_schedule_steps and --lr_schedule_gammas are required when --use_lr_schedule is set.")
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        parser.error("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")

# Validate FiLM arguments if strategy is film
if args.model_type == "deeposet" and args.encoding_strategy == "film":
    # Positional encoding must be enabled for FiLM
    # We can enforce this or rely on the check within DeepOSet.__init__
    pass # DeepOSet init handles this check

# hyper params
epochs = args.epochs
n_basis = args.n_basis
if args.device == "auto": # automatically choose
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device == "cuda" or args.device == "cpu": # use specificed device
    device = args.device
else: # use cuda device at this index
    device = f"cuda:{int(args.device)}"
seed = args.seed
load_path = args.load_path
model_type = args.model_type
dataset_type = args.dataset_type
nonlinear_datasets = ["MountainCar", "Elastic", "Darcy", "Heat", "LShaped", "Burger"]
transformation_type = "nonlinear" if args.dataset_type in nonlinear_datasets else "linear"
n_layers = args.n_layers
freeze_example_xs_train = not args.unfreeze_sensors

# POD is a special case, since it cant compute more eigen functions (Basis functions) then there are data points.
# 2Stage is likewise affected
if args.model_type in ["deeponet_pod", "deeponet_2stage"] and args.dataset_type == "Darcy" and n_basis > 40:
    print("WARNING: Darcy dataset has a maximum of 40 basis functions for DeepONet_POD, since the number of datapoints is 40. Setting n_basis to 40.")
    n_basis = 40

print(f"Training {model_type} on {transformation_type} {dataset_type} for {epochs} epochs, seed {seed}, with {n_basis} basis functions and {args.n_sensors} sensors.")

# generate logdir
if load_path is None:
    model_name_for_saving = f"{model_type}_{args.train_method}" if model_type in ["SVD", "Eigen", "matrix"] else model_type
    logdir = f"{args.logdir}/{dataset_type}/{model_name_for_saving}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# Determine if training source sensors should be frozen
freeze_example_xs_train = not args.unfreeze_sensors

# generate datasets
# Pass the training freeze flag and the testing flags to get_dataset
src_dataset, tgt_dataset, combined_dataset = get_dataset(
    dataset_type, test=False, model_type=model_type, n_sensors=args.n_sensors, device=device,
    freeze_example_xs_train=freeze_example_xs_train,
    test_variable_sensors=args.test_variable_sensors,
    no_copy_sensors_to_test=args.no_copy_sensors_to_test
)
testing_src_dataset, testing_tgt_dataset, testing_combined_dataset = get_dataset(
    dataset_type, test=True, model_type=model_type, n_sensors=args.n_sensors, device=device,
    freeze_example_xs_train=freeze_example_xs_train, # Pass the TRAIN freeze setting
    test_variable_sensors=args.test_variable_sensors, # Pass the test setting
    no_copy_sensors_to_test=args.no_copy_sensors_to_test # Pass the test setting
)

# Add a print statement to confirm the test dataset's freezing status
if args.model_type == "deeposet" or "deeponet" in args.model_type:
    if testing_combined_dataset.src_dataset.freeze_example_xs:
        print("Testing dataset source sensors (example_xs) are FROZEN.")
    else:
        print("Testing dataset source sensors (example_xs) are VARIABLE.")

# Copy source sensor locations from training to testing when appropriate
# Only copy if: source sensors are frozen for testing AND we're not testing variable sensors OR not copying sensors is disabled
if (testing_combined_dataset.src_dataset.freeze_example_xs and 
    (not args.test_variable_sensors or not args.no_copy_sensors_to_test)):
    # Make sure the training dataset has initialized its example_xs
    if combined_dataset.example_xs is None and combined_dataset.src_dataset.freeze_example_xs:
        # Force initialization by sampling
        example_xs, _, _, _, _ = combined_dataset.sample(device)
    
    # Now copy from training to testing
    if combined_dataset.example_xs is not None:
        if testing_combined_dataset.example_xs is None:
            testing_combined_dataset.example_xs = combined_dataset.example_xs
        
        if testing_combined_dataset.src_dataset.example_xs is None:
            testing_combined_dataset.src_dataset.example_xs = combined_dataset.src_dataset.example_xs
        
        print(f"Copied frozen source sensor locations from training to testing dataset.")
    else:
        print("Warning: Training dataset's sensors not initialized yet, cannot copy to testing dataset.")

# if using POD or 2stage, we need to copy the output sensors (Target queries 'xs')
# This logic remains the same as it concerns the target dataset's 'xs'
if args.model_type == "deeponet_pod" or args.model_type == "deeponet_2stage":
    # Check if the target dataset actually froze xs during its creation
    if combined_dataset.tgt_dataset.freeze_xs:
        testing_combined_dataset.tgt_dataset.frozen_xs = combined_dataset.tgt_dataset.frozen_xs
        testing_combined_dataset.frozen_xs = combined_dataset.frozen_xs # Assuming CombinedDataset also stores it
        print("Copying frozen target query locations (xs) from training to testing dataset.")
    else:
         print("Target query locations (xs) are not frozen for training, no copy needed for testing.")


# calculate hidden layer size based on approximate number of parameters
if args.model_type != "deeposet": # Calculate only if NOT deeposet
    hidden_size = get_hidden_layer_size(
        target_n_parameters=args.approximate_number_paramaters,
        n_layers=n_layers,
        src_input_space=src_dataset.input_size,
        src_output_space=src_dataset.output_size,
        tgt_input_space=tgt_dataset.input_size,
        tgt_output_space=tgt_dataset.output_size,
        n_sensors=combined_dataset.n_examples_per_sample,
        n_basis=n_basis,
        model_type=model_type, # Explicitly pass model_type by name
        transformation_type=transformation_type,
        dataset_type=dataset_type
    )
else:
    hidden_size = None # Not used directly for DeepOSet instantiation anymore

# create the model
if args.model_type == "SVD" or args.model_type == "Eigen":
    model = SVDEncoder(input_size_src=src_dataset.input_size,
                       output_size_src=src_dataset.output_size,
                       input_size_tgt=tgt_dataset.input_size,
                       output_size_tgt=tgt_dataset.output_size,
                       data_type="deterministic", # we dont support stochastic for now, though its possible.
                       n_basis=n_basis,
                       method=args.train_method,
                       use_eigen_decomp=(model_type=="Eigen"),
                       model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size}).to(device)
elif args.model_type == "matrix":
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
    model = {"src": src_model, "tgt": tgt_model}

    # optionally add neural network to transform between spaces for nonlinear operator
    if transformation_type == "nonlinear":
        transformation_input_size = src_model.n_basis if src_model is not None else src_dataset.output_size[0]
        layers = [torch.nn.Linear(transformation_input_size, hidden_size),torch.nn.ReLU()]
        for layer in range(n_layers - 2):
            layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hidden_size, tgt_model.n_basis)]
        a_model = torch.nn.Sequential(*layers).to(device)
        model["A"] = a_model
        opt = torch.optim.Adam(model["A"].parameters(), lr=1e-3)
    else:
        model["A"] = torch.rand(tgt_model.n_basis, src_model.n_basis).to(device)
elif args.model_type == "deeponet_cnn":
    model = DeepONet_CNN(input_size_tgt=tgt_dataset.input_size[0],
                        output_size_tgt=tgt_dataset.output_size[0],
                        input_size_src=src_dataset.input_size[0],
                        output_size_src=src_dataset.output_size[0],
                        n_input_sensors=combined_dataset.n_examples_per_sample,
                        p=n_basis,
                        n_layers=n_layers,
                        hidden_size=hidden_size,
                        ).to(device)
elif args.model_type == "deeponet_pod":
    model = DeepONet_POD(input_size_tgt=tgt_dataset.input_size[0],
                        output_size_tgt=tgt_dataset.output_size[0],
                        input_size_src=src_dataset.input_size[0],
                        output_size_src=src_dataset.output_size[0],
                        n_input_sensors=combined_dataset.n_examples_per_sample,
                        p=n_basis,
                        n_layers=n_layers,
                        hidden_size=hidden_size,
                        ).to(device)
    # make the tgt_dataset us a bunch of functions for this calculation only
    total_n_functions = tgt_dataset.n_functions
    n_f_per_sample = tgt_dataset.n_functions_per_sample
    tgt_dataset.n_functions_per_sample = total_n_functions if type(total_n_functions) == int else 999

    model.compute_POD(tgt_dataset)

    # reset tgt dataset
    tgt_dataset.n_functions_per_sample = n_f_per_sample

elif args.model_type == "deeponet_2stage":
    # consists of basis functions at the target
    # and a deeponet style branch to compute the coefficients
    tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                                output_size=tgt_dataset.output_size,
                                data_type=tgt_dataset.data_type,
                                n_basis=n_basis,
                                method=args.train_method,
                                regularization_parameter=100.0,
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                ).to(device)
    src_basis = None
    T = torch.rand(n_basis, n_basis).to(device)

    # create A matrix, which is basically the branch for deeponet
    transformation_input_size = combined_dataset.n_examples_per_sample * src_dataset.output_size[0]
    layers = [torch.nn.Linear(transformation_input_size, hidden_size),torch.nn.ReLU()]
    for layer in range(n_layers - 2):
        layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
    layers += [torch.nn.Linear(hidden_size, tgt_model.n_basis)]
    a_model = torch.nn.Sequential(*layers).to(device)
    A = a_model
    opt = torch.optim.Adam(A.parameters(), lr=1e-3)
    model = {"src": src_basis, "tgt": tgt_model, "A": A, "T": T}

elif args.model_type == "deeponet_2stage_cnn":
    # consists of basis functions at the target
    # and a deeponet style branch to compute the coefficients
    tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                                output_size=tgt_dataset.output_size,
                                data_type=tgt_dataset.data_type,
                                n_basis=n_basis,
                                method=args.train_method,
                                regularization_parameter=100.0,
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                ).to(device)
    src_basis = None
    T = torch.rand(n_basis, n_basis).to(device)

    # create A matrix, which is basically the branch for deeponet
    A = DeepONet_2Stage_CNN_branch(input_size_tgt=tgt_dataset.input_size[0],
                                      output_size_tgt=tgt_dataset.output_size[0],
                                      input_size_src=src_dataset.input_size[0],
                                      output_size_src=src_dataset.output_size[0],
                                      n_input_sensors=combined_dataset.n_examples_per_sample,
                                      p=n_basis,
                                      n_layers=n_layers,
                                      hidden_size=hidden_size,
                                      ).to(device)
    opt = torch.optim.Adam(A.parameters(), lr=1e-3)
    model = {"src": src_basis, "tgt": tgt_model, "A": A, "T": T}


elif args.model_type == "deeponet":
    # Conditionally set schedule parameters based on the flag
    schedule_steps = args.lr_schedule_steps if args.use_lr_schedule else None
    schedule_gammas = args.lr_schedule_gammas if args.use_lr_schedule else None

    model = DeepONet(input_size_tgt=tgt_dataset.input_size[0],
                     output_size_tgt=tgt_dataset.output_size[0],
                     input_size_src=src_dataset.input_size[0],
                     output_size_src=src_dataset.output_size[0],
                     n_input_sensors=combined_dataset.n_examples_per_sample,
                     p=n_basis,
                     n_layers=n_layers,
                     hidden_size=hidden_size,
                     # Pass the schedule parameters (could be None)
                     lr_schedule_steps=schedule_steps,
                     lr_schedule_gammas=schedule_gammas
                     ).to(device)
elif args.model_type == "deeposet":
    # Conditionally set schedule parameters based on the flag
    schedule_steps = args.lr_schedule_steps if args.use_lr_schedule else None
    schedule_gammas = args.lr_schedule_gammas if args.use_lr_schedule else None

    # Determine if positional encoding should be used based on strategy AND type
    use_pos_encoding = args.encoding_strategy in ['concatenate', 'film'] and args.pos_encoding_type != 'skip'

    model = DeepOSet(
        input_size_src=src_dataset.input_size[0],
        output_size_src=src_dataset.output_size[0],
        input_size_tgt=tgt_dataset.input_size[0],
        output_size_tgt=tgt_dataset.output_size[0],
        p=n_basis, # Using n_basis as the latent dimension 'p'
        # Network sizes from args
        trunk_hidden_size=args.trunk_hidden_size,
        phi_hidden_size=args.phi_hidden_size,
        rho_hidden_size=args.rho_hidden_size,
        phi_output_size=args.phi_output_size,
        # Encoding strategy and related parameters from args
        encoding_strategy=args.encoding_strategy,
        use_positional_encoding=use_pos_encoding, # Pass updated flag
        pos_encoding_type=args.pos_encoding_type, # Pass from args (can be 'skip')
        pos_encoding_dim=args.pos_encoding_dim,   # Pass from args
        pos_encoding_max_freq=args.pos_encoding_max_freq, # Pass from args
        film_modulation_dim=args.film_modulation_dim, # Pass from args (can be None)
        # LR schedule parameters from args
        lr_schedule_steps=schedule_steps,
        lr_schedule_gammas=schedule_gammas,
        # NEW
        aggregation_type=args.aggregation_type,
        attention_n_tokens=args.attention_n_tokens,
    ).to(device)
else:
    raise ValueError(f"Unknown model type: {args.model_type}")

# get number of parameters
n_params = get_num_parameters(model)
# The prediction function needs the individual hidden sizes for DeepOSet now,
# or we can skip the check for DeepOSet if predict_number_params isn't updated.
# Let's skip the check for simplicity for now.
if args.model_type != "deeposet":
    # For other models, hidden_size was calculated and can be used
    predict_n_params = predict_number_params(model_type, combined_dataset.n_examples_per_sample, n_basis, hidden_size, n_layers, src_dataset.input_size, src_dataset.output_size, tgt_dataset.input_size, tgt_dataset.output_size, transformation_type, dataset_type)
    assert n_params == predict_n_params, f"Number of parameters is not consistent for {model_type}, expected {predict_n_params}, got {n_params}."
else:
    # For DeepOSet, we used direct hidden sizes.
    # We could update predict_number_params or just print the count.
    print(f"DeepOSet model created with {n_params} parameters.")
    # Optionally, update predict_number_params to accept phi/rho/trunk sizes
    # predict_n_params = predict_number_params_deeposet(...) # Hypothetical updated function
    # assert n_params == predict_n_params, "..."

# writes all parameters and saves them
params = {"seed": seed,
          "n_sensors": args.n_sensors,
          "n_basis": n_basis, # This is 'p' for DeepONet/DeepOSet
          "n_params": n_params,
          "n_layers": n_layers, # Trunk layers for DeepONet/DeepOSet
          # Store the specific hidden sizes used
          "phi_hidden_size": args.phi_hidden_size if args.model_type == "deeposet" else (hidden_size if args.model_type != "matrix" else None),
          "rho_hidden_size": args.rho_hidden_size if args.model_type == "deeposet" else (hidden_size if args.model_type != "matrix" else None),
          "trunk_hidden_size": args.trunk_hidden_size if args.model_type == "deeposet" else (hidden_size if args.model_type != "matrix" else None),
          # "approximate_number_parameters": args.approximate_number_paramaters, # Less relevant now for DeepOSet
          "model_type": model_type,
          "train_method": args.train_method if model_type in ["SVD", "Eigen", "matrix"] else None,
          "dataset_type": dataset_type,
          "transformation_type": transformation_type,
          "device": device,
          "logdir": logdir,
          "epochs": epochs,
          # Add DeepOSet specific positional encoding params
          "pos_encoding_type": args.pos_encoding_type if args.model_type == "deeposet" else None,
          "pos_encoding_dim": args.pos_encoding_dim if args.model_type == "deeposet" else None,
          "pos_encoding_max_freq": args.pos_encoding_max_freq if args.model_type == "deeposet" and args.pos_encoding_type == 'sinusoidal' else None,
          # Add DeepOSet encoding strategy params
          "encoding_strategy": args.encoding_strategy if args.model_type == "deeposet" else None,
          "film_modulation_dim": args.film_modulation_dim if args.model_type == "deeposet" and args.encoding_strategy == 'film' else None,
          "aggregation_type": args.aggregation_type if args.model_type == "deeposet" else None,
          "attention_n_tokens": args.attention_n_tokens if (args.model_type == "deeposet" and args.aggregation_type=="attention") else None,
          }
# Add LR schedule info only if the model has it and schedule was used
if hasattr(model, 'initial_lr'):
    params["initial_lr"] = model.initial_lr
if hasattr(model, 'lr_schedule_steps') and model.lr_schedule_steps is not None:
     params["lr_schedule_steps"] = str(model.lr_schedule_steps)
     # Check for gammas specifically
     if hasattr(model, 'lr_schedule_gammas') and model.lr_schedule_gammas is not None:
         params["lr_schedule_gammas"] = str(model.lr_schedule_gammas)

os.makedirs(logdir, exist_ok=True)
# Use model._param_string() if available (like in DeepOSet) for consistency
if hasattr(model, '_param_string') and callable(model._param_string):
    params_from_model = model._param_string()
    # Update the main params dict, potentially overwriting some if names clash
    # but ensuring model's internal view is saved.
    params.update(params_from_model)
    # Convert all values to string for saving, as _param_string already does
    params = {k: str(v) for k, v in params.items()}
else:
    # Fallback for models without _param_string
    params = {k: str(v) for k, v in params.items() if v is not None} # Ensure None is not saved directly

torch.save(params, f"{logdir}/params.pth")

# Save the command-line arguments as a JSON file
args_dict = vars(args) # Convert argparse Namespace to dictionary
args_save_path = f"{logdir}/args.json"
with open(args_save_path, 'w') as f:
    json.dump(args_dict, f, indent=4)
print(f"Saved command-line arguments to {args_save_path}")


# train or load a model
if load_path is not None: # load models
    if args.model_type == "matrix":
        if model["src"] is not None:
            model["src"].load_state_dict(torch.load(f"{logdir}/src_model.pth", weights_only=True))
        model["tgt"].load_state_dict(torch.load(f"{logdir}/tgt_model.pth", weights_only=True))
        if transformation_type == "linear":
            model["A"] = torch.load(f"{logdir}/A.pth", weights_only=True)
        else:
            model["A"].load_state_dict(torch.load(f"{logdir}/A.pth", weights_only=True))
    elif args.model_type in ["deeponet_2stage", "deeponet_2stage_cnn"]:
        model["tgt"].load_state_dict(torch.load(f"{logdir}/tgt_model.pth", weights_only=True))
        model["A"].load_state_dict(torch.load(f"{logdir}/A.pth", weights_only=True))
    else:
        model.load_state_dict(torch.load(f"{logdir}/model.pth", weights_only=True))
else: # train models
    # create callbacks
    if args.model_type == "matrix":
        callback = TensorboardCallback(logdir=logdir, prefix="source")
        callback2 = TensorboardCallback(tensorboard=callback.tensorboard, prefix="target") # this logs to the same tensorboard but with a different prefix
    else:
        callback = TensorboardCallback(logdir) # this one logs training data

    # train and test occasionally
    if args.model_type == "matrix" and transformation_type == "linear":
        model["A"] = compute_A(model["src"], model["tgt"], combined_dataset, device, args.train_method, callback)
    test(model, testing_combined_dataset, callback2 if args.model_type == "matrix" else callback, transformation_type, args.train_method, args.model_type)
    num_tests = 100 if epochs > 0 else 0
    for iteration in trange(num_tests):

        # training step
        if args.model_type == "matrix":
            if model["src"] is not None:
                model["src"].train_model(src_dataset, epochs=epochs//num_tests, callback=callback, progress_bar=False)
            model["tgt"].train_model(tgt_dataset, epochs=epochs//num_tests, callback=callback2, progress_bar=False)
            if transformation_type == "linear":
                model["A"] = compute_A(model["src"], model["tgt"], combined_dataset, device, args.train_method, callback)
            else:
                model["A"], opt, _ = train_nonlinear_transformation(model["A"], opt, model["src"], model["tgt"], args.train_method, combined_dataset, epochs//num_tests, callback, model_type)
        elif args.model_type in ["deeponet_2stage", "deeponet_2stage_cnn"]:
            model["tgt"].train_model(tgt_dataset, epochs=epochs//num_tests, callback=callback, progress_bar=False)
            model["A"], opt, model["T"] = train_nonlinear_transformation(model["A"], opt, model["src"],  model["tgt"], args.train_method, combined_dataset, epochs//num_tests, callback, model_type)

        else:
            model.train_model(combined_dataset, epochs=epochs//num_tests, callback=callback, progress_bar=False)

        # testing step.
        test(model, testing_combined_dataset, callback2 if args.model_type == "matrix" else callback, transformation_type, args.train_method, args.model_type)



    # save the model
    if args.model_type == "matrix":
        if model["src"] is not None:
            torch.save(model["src"].state_dict(), f"{logdir}/src_model.pth")
        torch.save(model["tgt"].state_dict(), f"{logdir}/tgt_model.pth")
        if transformation_type == "linear":
            torch.save(model["A"], f"{logdir}/A.pth")
        else:
            torch.save(model["A"].state_dict(), f"{logdir}/A.pth")
    elif args.model_type in ["deeponet_2stage", "deeponet_2stage_cnn"]:
        torch.save(model["tgt"].state_dict(), f"{logdir}/tgt_model.pth")
        torch.save(model["A"].state_dict(), f"{logdir}/A.pth")
        torch.save(model["T"], f"{logdir}/T.pth")
    else:
        torch.save(model.state_dict(), f"{logdir}/model.pth")

##############   Evaluate    ###################
with torch.no_grad():
    # fetch the correct plotting functions
    if args.dataset_type == "QuadraticSin":
        plot_source = plot_source_quadratic
        plot_target = plot_target_sin
        plot_transformation = plot_transformation_quadratic_sin
    elif args.dataset_type == "Derivative":
        plot_source = plot_source_cubic
        plot_target = plot_target_cubic_derivative
        plot_transformation = plot_transformation_derivative
    elif args.dataset_type == "Integral":
        plot_source = plot_source_quadratic
        plot_target = plot_target_quadratic_integral
        plot_transformation = plot_transformation_integral
    elif args.dataset_type == "MountainCar":
        plot_source = plot_source_mountain_car
        plot_target = plot_target_mountain_car
        plot_transformation = plot_transformation_mountain_car
    elif args.dataset_type == "Elastic":
        plot_source = plot_source_boundary_force
        plot_target = plot_target_boundary
        plot_transformation = plot_transformation_elastic
    elif args.dataset_type == "Darcy":
        plot_source = plot_source_darcy
        plot_target = plot_target_darcy
        plot_transformation = plot_transformation_darcy
    elif args.dataset_type == "Heat":
        plot_source = plot_source_heat
        plot_target = plot_target_heat
        plot_transformation = plot_transformation_heat
    elif args.dataset_type == "LShaped":
        plot_source = plot_source_L
        plot_target = plot_target_L
        plot_transformation = plot_transformation_L
    elif args.dataset_type == "Burger":
        plot_source = plot_source_burger
        plot_target = plot_target_burger
        plot_transformation = plot_transformation_burger
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


    # plot src and target fit, if using SVD, Eigen, or Matrix
    if args.model_type == "SVD" or args.model_type == "Eigen" or args.model_type == "matrix":

        if dataset_type != "Heat":
            # get data
            example_xs, example_ys, xs, ys, info = src_dataset.sample(device, plot_only=True)
            info["model_type"] = f"{model_type}_{args.train_method}" if ("deeponet" not in model_type)else model_type
            # mountain car plot needs a 2d grid instead of the random data, for plotting purposes.
            if args.dataset_type == "MountainCar":
                x_1 = torch.linspace(-1.2, 0.6, 100)
                x_2 = torch.linspace(-0.07, 0.07, 100)
                x_1, x_2 = torch.meshgrid(x_1, x_2)
                xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
                xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
                ys = src_dataset.compute_outputs(info, xs)
                xs, ys = xs.to(device), ys.to(device)
            elif args.dataset_type == "Fluid":
                x_1 = src_dataset.xx1
                x_2 = src_dataset.xx2
                x_1, x_2 = torch.meshgrid(x_1, x_2)
                xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
                xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
                ys = src_dataset.ys[info["function_indicies"]]
                xs, ys = xs.to(device), ys.to(device)

            if args.model_type == "matrix":
                y_hats = model["src"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)
            elif args.model_type == "SVD" or args.model_type == "Eigen":
                y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="source", prediction_dataset="source")

            # plot source domain
            plot_source(xs, ys, y_hats, info, logdir)

        # get data
        example_xs, example_ys, xs, ys, info = tgt_dataset.sample(device, plot_only=True)
        info["model_type"] = f"{model_type}_{args.train_method}" if ("deeponet" not in model_type)else model_type
        if args.dataset_type == "Fluid":
            x_1 = tgt_dataset.xx1
            x_2 = tgt_dataset.xx2
            x_1, x_2 = torch.meshgrid(x_1, x_2)
            xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
            xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
            ys = tgt_dataset.ys[info["function_indicies"]]
            xs, ys = xs.to(device), ys.to(device)
        elif args.dataset_type == "Heat":
            function_indicies = info["function_indicies"]
            xs = tgt_dataset.xs[function_indicies]
            ys = tgt_dataset.ys[function_indicies]
            times = [0, 20, 40, 60]
            size = 99*99
            new_xs, new_ys = [], []
            for time in times:
                temp_xs = xs[:, size * time: size * (time + 1)]
                temp_ys = ys[:, size * time: size * (time + 1)]
                new_xs.append(temp_xs)
                new_ys.append(temp_ys)
            xs = torch.cat(new_xs, dim=1).to(device)
            ys = torch.cat(new_ys, dim=1).to(device)





        if args.model_type == "matrix":
            y_hats = model["tgt"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)
        else:
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="target", prediction_dataset="target")

        # plot target domain
        plot_target(xs, ys, y_hats, info, logdir)


    # plot transformation for all model types
    example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device)
    info["model_type"] = f"{model_type}_{args.train_method}" if model_type in ["SVD", "Eigen", "matrix"] else model_type

    # mountain car plot needs a 2d grid instead of the random data, for plotting purposes.
    if args.dataset_type == "MountainCar":
        x_1 = torch.linspace(-1.2, 0.6, 100)
        x_2 = torch.linspace(-0.07, 0.07, 100)
        x_1, x_2 = torch.meshgrid(x_1, x_2)
        grid = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
        grid = grid.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
        grid_outs = src_dataset.compute_outputs(info, grid)
        grid, grid_outs = grid.to(device), grid_outs.to(device)
    elif args.dataset_type == "Fluid":
        x_1 = src_dataset.xx1
        x_2 = src_dataset.xx2
        x_1, x_2 = torch.meshgrid(x_1, x_2)
        grid = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
        grid = grid.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
        grid_outs = src_dataset.ys[info["function_indicies"]]
        grid, grid_outs = grid.to(device), grid_outs.to(device)
        xs = grid
        ys = tgt_dataset.ys[info["function_indicies"]]
    elif args.dataset_type == "Heat":
        function_indicies = info["function_indicies"]
        all_xs = testing_combined_dataset.tgt_dataset.xs[function_indicies]
        all_ys = testing_combined_dataset.tgt_dataset.ys[function_indicies]

        # get subset we want to plot        
        xs = all_xs[:, 49::99, :]
        ys = all_ys[:, 49::99, :]
        grid = example_xs
        grid_outs =  example_ys
    elif args.dataset_type == "Burger":
        function_indicies = info["function_indicies"]
        xs = testing_combined_dataset.tgt_dataset.xs.repeat(10, 1, 1)
        ys = testing_combined_dataset.tgt_dataset.ys[function_indicies]
        grid = example_xs
        grid_outs = example_ys

    else:
        grid = example_xs
        grid_outs = example_ys


    # first compute example y_hats for the three model types that can do this.
    if args.model_type == "SVD" or args.model_type == "Eigen" or (args.model_type == "matrix" and dataset_type != "Heat"):
        if args.model_type == "matrix":
            example_y_hats = model["src"].predict_from_examples(example_xs, example_ys, grid, method=args.train_method)
        else:
            example_y_hats = model.predict_from_examples(example_xs, example_ys, grid, method=args.train_method, representation_dataset="source", prediction_dataset="source")
    else:
        example_y_hats = None

    # next compute y_hats for all models
    if args.model_type == "matrix":
        if type(combined_dataset.src_dataset) == HeatSrcDataset:
            rep = example_ys[:, 0, :]
        else:
            rep, _ = model["src"].compute_representation(example_xs, example_ys, method=args.train_method)


        if transformation_type == "linear":
            rep = rep @ model["A"].T
        else:
            rep = model["A"](rep)
        y_hats = model["tgt"].predict(xs, rep)
    elif args.model_type == "SVD" or args.model_type == "Eigen":
        y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="source", prediction_dataset="target")

    elif args.model_type == "deeponet_2stage":
        tgt_Cs_hat = (model["T"] @ model["A"](example_ys.reshape(example_ys.shape[0], -1)).T).T
        y_hats = model["tgt"].predict(xs, tgt_Cs_hat)
    elif args.model_type == "deeponet_2stage_cnn":
        tgt_Cs_hat = (model["T"] @ model["A"](example_ys).T).T
        y_hats = model["tgt"].predict(xs, tgt_Cs_hat)

    else: # deeponet*, deeposet
        y_hats = model.forward(example_xs, example_ys, xs)

    # plot
    if not (args.dataset_type in ["Heat", "Burger"] and args.model_type == "deeponet_pod"): # POD cannot be called on new inputs, so it cannot plot.
        plot_transformation(grid, grid_outs, example_y_hats, xs, ys, y_hats, info, logdir)


# plot transformation for all model types using TESTING data
print("\nGenerating transformation plot using TESTING dataset...")
example_xs_test, example_ys_test, xs_test, ys_test, info_test = testing_combined_dataset.sample(device) # Use testing dataset
info_test["model_type"] = f"{model_type}_{args.train_method}" if model_type in ["SVD", "Eigen", "matrix"] else model_type
plot_test_example_xs_id = id(example_xs_test)
print(f"  Plotting Test Data: example_xs_test ID: {plot_test_example_xs_id}, Shape: {example_xs_test.shape}, First val: {example_xs_test.flatten()[0].item():.4f}")

# --- Predict using TESTING data ---
example_y_hats_test = None
y_hats_test = None
with torch.no_grad():
    if model_type == "matrix":
        if type(testing_combined_dataset.src_dataset) == HeatSrcDataset:
             src_Cs_test = example_ys_test[:, 0, :]
        else:
             src_Cs_test, _ = model["src"].compute_representation(example_xs_test, example_ys_test, method=args.train_method)
        if transformation_type == "linear":
            tgt_Cs_hat_test = src_Cs_test @ model["A"].T
        else:
            tgt_Cs_hat_test = model["A"](src_Cs_test)
        y_hats_test = model["tgt"].predict(xs_test, tgt_Cs_hat_test)
        # Optionally predict source reconstruction if src_model exists
        if model["src"] is not None:
             example_y_hats_test = model["src"].predict(example_xs_test, src_Cs_test)

    elif model_type == "SVD" or model_type == "Eigen":
        y_hats_test = model.predict_from_examples(example_xs_test, example_ys_test, xs_test, method=args.train_method, representation_dataset="source", prediction_dataset="target")
        # SVD/Eigen might not have a direct source reconstruction, set example_y_hats_test if needed/possible
        # example_y_hats_test = model.predict_from_examples(example_xs_test, example_ys_test, example_xs_test, ...) # Example

    elif model_type == "deeponet_2stage":
        tgt_Cs_hat_test = (model["T"] @ model["A"](example_ys_test.reshape(example_ys_test.shape[0], -1)).T).T
        y_hats_test = model["tgt"].predict(xs_test, tgt_Cs_hat_test)
        # Predict source reconstruction if needed
        # example_y_hats_test = ... # Requires source model prediction logic if applicable

    elif model_type == "deeponet_2stage_cnn":
        tgt_Cs_hat_test = (model["T"] @ model["A"](example_ys_test).T).T
        y_hats_test = model["tgt"].predict(xs_test, tgt_Cs_hat_test)
        # Predict source reconstruction if needed
        # example_y_hats_test = ... # Requires source model prediction logic if applicable

    else: # deeponet*, deeposet
        y_hats_test = model.forward(example_xs_test, example_ys_test, xs_test)
        # Predict source reconstruction (if the model supports/needs it for plotting)
        # example_y_hats_test = model.forward(example_xs_test, example_ys_test, example_xs_test) # Example

# --- Call Plotting Function for TESTING data ---
# mountain car plot needs a 2d grid instead of the random data, for plotting purposes.
if args.dataset_type == "MountainCar":
    plot_transformation_mountain_car(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "QuadraticSin":
    plot_transformation_quadratic_sin(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "Derivative":
    plot_transformation_derivative(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "Integral":
    plot_transformation_integral(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "Elastic":
    plot_transformation_elastic(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "Darcy":
    plot_transformation_darcy(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "Heat":
    plot_transformation_heat(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "LShaped":
    plot_transformation_L(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)
elif args.dataset_type == "Burger":
    plot_transformation_burger(example_xs_test, example_ys_test, example_y_hats_test, xs_test, ys_test, y_hats_test, info_test, logdir)


# --- NEW SECTION: Plot transformation using TRAINING data ---
print("\nGenerating transformation plot using TRAINING dataset...")
# Sample training data AGAIN for plotting
example_xs_train, example_ys_train, xs_train, ys_train, info_train = combined_dataset.sample(device)
info_train["model_type"] = f"{model_type}_{args.train_method}" if model_type in ["SVD", "Eigen", "matrix"] else model_type
plot_train_example_xs_id = id(example_xs_train)
print(f"  Plotting Train Data: example_xs_train ID: {plot_train_example_xs_id}, Shape: {example_xs_train.shape}, First val: {example_xs_train.flatten()[0].item():.4f}")

# --- Predict using TRAINING data ---
example_y_hats_train = None
y_hats_train = None
with torch.no_grad():
    # Replicate prediction logic using _train variables
    if model_type == "matrix":
        if type(combined_dataset.src_dataset) == HeatSrcDataset: # Check training dataset type
             src_Cs_train = example_ys_train[:, 0, :]
        else:
             src_Cs_train, _ = model["src"].compute_representation(example_xs_train, example_ys_train, method=args.train_method)
        if transformation_type == "linear":
            tgt_Cs_hat_train = src_Cs_train @ model["A"].T
        else:
            tgt_Cs_hat_train = model["A"](src_Cs_train)
        y_hats_train = model["tgt"].predict(xs_train, tgt_Cs_hat_train)
        if model["src"] is not None:
             example_y_hats_train = model["src"].predict(example_xs_train, src_Cs_train)

    elif model_type == "SVD" or model_type == "Eigen":
        y_hats_train = model.predict_from_examples(example_xs_train, example_ys_train, xs_train, method=args.train_method, representation_dataset="source", prediction_dataset="target")
        # example_y_hats_train = ... # Predict source if needed

    elif model_type == "deeponet_2stage":
        tgt_Cs_hat_train = (model["T"] @ model["A"](example_ys_train.reshape(example_ys_train.shape[0], -1)).T).T
        y_hats_train = model["tgt"].predict(xs_train, tgt_Cs_hat_train)
        # example_y_hats_train = ... # Predict source if needed

    elif model_type == "deeponet_2stage_cnn":
        tgt_Cs_hat_train = (model["T"] @ model["A"](example_ys_train).T).T
        y_hats_train = model["tgt"].predict(xs_train, tgt_Cs_hat_train)
        # example_y_hats_train = ... # Predict source if needed

    else: # deeponet*, deeposet
        y_hats_train = model.forward(example_xs_train, example_ys_train, xs_train)
        # example_y_hats_train = model.forward(example_xs_train, example_ys_train, example_xs_train) # Predict source if needed

# --- Call Plotting Function for TRAINING data ---
# Create a subdirectory for these plots
train_plot_logdir = os.path.join(logdir, "train_sensor_plots")
os.makedirs(train_plot_logdir, exist_ok=True)

# Call the same plotting functions but with _train data and the new logdir
if args.dataset_type == "MountainCar":
    plot_transformation_mountain_car(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "QuadraticSin":
    plot_transformation_quadratic_sin(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "Derivative":
    plot_transformation_derivative(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "Integral":
    plot_transformation_integral(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "Elastic":
    plot_transformation_elastic(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "Darcy":
    plot_transformation_darcy(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "Heat":
    plot_transformation_heat(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "LShaped":
    plot_transformation_L(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)
elif args.dataset_type == "Burger":
    plot_transformation_burger(example_xs_train, example_ys_train, example_y_hats_train, xs_train, ys_train, y_hats_train, info_train, train_plot_logdir)

print(f"Training sensor plots saved in: {train_plot_logdir}")

# --- End of Script ---



