import deepxde as dde
import os # Make sure os is imported early
import numpy as np
import matplotlib.pyplot as plt
# Import PyTorch
import torch
import torch.nn as nn
from datetime import datetime # Import datetime for timestamp
import torch.optim as optim # Import optimizer
from torch.utils.data import Dataset, DataLoader # Import PyTorch data utilities

# --- Import the new DeepOSets module ---
from deeposets_module import PyTorchDeepSetsBranch, PyTorchTrunkNet, CustomDeepONet
# --- End Import ---

# --- Debug Backend ---
print(f"Environment DDE_BACKEND: {os.environ.get('DDE_BACKEND')}")
print(f"DeepXDE selected backend: {dde.backend.backend_name}")
# --- End Debug ---

# Define ranges for coefficients and input
A_RANGE = (-1.5, 1.5)
B_RANGE = (-1.5, 1.5)
C_RANGE = (-1.5, 1.5)
INPUT_RANGE = (-10, 10)
N_POINTS = 100 # Number of points to evaluate each function at (for trunk net and output)
# N_SENSORS defined later or via args/grid search

def quadratic_func(coeffs, x):
    """Calculates y = ax^2 + bx + c. Uses NumPy."""
    a, b, c = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
    # Ensure x is treated correctly for broadcasting
    if x.ndim == 1:
        x_t = x.reshape(1, -1) # Shape (1, n_points)
    elif x.ndim == 2 and x.shape[1] == 1:
        x_t = x.T # Shape (1, n_points)
    else: # Assume shape is already (1, n_points) or similar
        x_t = x

    # Broadcasting: (n_funcs, 1) * (1, n_points) -> (n_funcs, n_points)
    return a * (x_t**2) + b * x_t + c

def get_data(n_train_funcs, n_test_funcs, n_points, x_sensors): # Accept x_sensors
    """
    Generates training and testing data using NumPy.
    Branch input: Function values at fixed sensor locations.
    Trunk input: Grid points for evaluation.
    Receives sensor locations x_sensors.
    """
    total_funcs = n_train_funcs + n_test_funcs
    n_sensors = x_sensors.shape[0] # Get n_sensors from input array

    # Generate coefficients
    a_coeffs = np.random.uniform(A_RANGE[0], A_RANGE[1], size=(total_funcs, 1))
    b_coeffs = np.random.uniform(B_RANGE[0], B_RANGE[1], size=(total_funcs, 1))
    c_coeffs = np.random.uniform(C_RANGE[0], C_RANGE[1], size=(total_funcs, 1))
    coeffs = np.hstack((a_coeffs, b_coeffs, c_coeffs)).astype(np.float32)

    # Calculate function values at sensor locations (branch input)
    # Use the provided x_sensors (ensure it's numpy if quadratic_func expects it)
    x_sensors_np = x_sensors if isinstance(x_sensors, np.ndarray) else x_sensors.numpy()
    y_sensors = quadratic_func(coeffs, x_sensors_np).astype(np.float32)

    # Generate grid points for trunk input and output evaluation
    # Shape: (n_points, 1)
    grid = np.linspace(INPUT_RANGE[0], INPUT_RANGE[1], n_points).reshape(-1, 1).astype(np.float32)

    # Calculate function values at grid points (output)
    # Shape: (total_funcs, n_points)
    y_data = quadratic_func(coeffs, grid).astype(np.float32)

    # Split into train/test
    x_branch_train = y_sensors[:n_train_funcs]
    x_branch_test = y_sensors[n_train_funcs:]
    y_train = y_data[:n_train_funcs]
    y_test = y_data[n_train_funcs:]
    test_coeffs = coeffs[n_train_funcs:]

    # Format for deepxde's TripleCartesianProd
    x_train = (x_branch_train, grid)
    x_test = (x_branch_test, grid)

    # Return grid and test_coeffs, x_sensors is no longer generated here
    return x_train, y_train, x_test, y_test, test_coeffs, grid

# Rename and modify generate_varied_sensor_data
def generate_batch_with_varied_sensors(batch_size, n_sensors_per_func, grid_points):
    """
    Generates a BATCH of data where each function is sampled at different,
    randomly chosen sensor locations. Suitable for PyTorch training loop.

    Args:
        batch_size: Number of functions in the batch.
        n_sensors_per_func: The number of sensor points for each function.
        grid_points: The grid points (numpy array, shape (n_points, 1)) for evaluating the true function.

    Returns:
        tuple: (batch_branch_vals, batch_branch_locs, batch_y_true)
            - batch_branch_vals: Numpy array [shape (batch_size, n_sensors_per_func)]
            - batch_branch_locs: Numpy array [shape (batch_size, n_sensors_per_func, 1)]
            - batch_y_true: Numpy array [shape (batch_size, n_points)] - true values on grid
    """
    # Pre-allocate arrays for efficiency
    n_points = grid_points.shape[0]
    batch_branch_vals = np.zeros((batch_size, n_sensors_per_func), dtype=np.float32)
    batch_branch_locs = np.zeros((batch_size, n_sensors_per_func, 1), dtype=np.float32)
    batch_y_true = np.zeros((batch_size, n_points), dtype=np.float32)

    # Generate coefficients for all functions in the batch at once
    a_coeffs = np.random.uniform(A_RANGE[0], A_RANGE[1], size=(batch_size, 1))
    b_coeffs = np.random.uniform(B_RANGE[0], B_RANGE[1], size=(batch_size, 1))
    c_coeffs = np.random.uniform(C_RANGE[0], C_RANGE[1], size=(batch_size, 1))
    batch_coeffs = np.hstack((a_coeffs, b_coeffs, c_coeffs)).astype(np.float32)

    for i in range(batch_size):
        coeffs_i = batch_coeffs[i:i+1, :] # Keep shape (1, 3) for quadratic_func

        # Generate UNIQUE, RANDOM sensor locations for this function
        x_sensors_i = np.sort(np.random.uniform(INPUT_RANGE[0], INPUT_RANGE[1], size=(n_sensors_per_func, 1))).astype(np.float32)

        # Calculate sensor values at these locations
        y_sensors_i = quadratic_func(coeffs_i, x_sensors_i).flatten().astype(np.float32) # Shape (n_sensors_per_func,)

        # Calculate true function values on the full grid
        y_grid_i = quadratic_func(coeffs_i, grid_points).flatten().astype(np.float32) # Shape (n_points,)

        batch_branch_vals[i, :] = y_sensors_i
        batch_branch_locs[i, :, :] = x_sensors_i
        batch_y_true[i, :] = y_grid_i

    # We don't need to return coeffs for standard training
    return batch_branch_vals, batch_branch_locs, batch_y_true

# --- Custom PyTorch Dataset (Optional but good practice) ---
class VariedSensorDataset(Dataset):
    def __init__(self, n_funcs, n_sensors_per_func, grid_points):
        self.n_funcs = n_funcs # Total number of functions in the dataset epoch
        self.n_sensors_per_func = n_sensors_per_func
        self.grid_points = grid_points
        # In this case, we generate data on the fly in __getitem__
        # For large datasets, pre-generation might be better

    def __len__(self):
        return self.n_funcs

    def __getitem__(self, idx):
        # Generate a single sample (equivalent to a batch of size 1)
        # Note: It's often more efficient to generate batches directly outside the Dataset
        #       if generation is fast, as done in the main loop below.
        #       This Dataset class is more illustrative here.
        vals, locs, true_y = generate_batch_with_varied_sensors(
            batch_size=1,
            n_sensors_per_func=self.n_sensors_per_func,
            grid_points=self.grid_points
        )
        # Remove the batch dimension added by the generator
        return vals.squeeze(0), locs.squeeze(0), true_y.squeeze(0)

# --- Modify plot_results to accept the PyTorch net directly ---
# (Signature changes slightly: accepts 'net' instead of 'model')
def plot_results(net, test_data, true_values, coefficients, grid, log_dir, plot_filename, title_prefix, n_plots=9):
    """
    Plots predictions vs true values for a subset of test functions.
    Accepts the PyTorch network directly.
    Can handle both fixed and varied sensor location test sets.
    """
    # Detect if the input data uses varied sensors
    # Varied data is now passed as (list_of_vals, list_of_locs, grid_tensor)
    is_varied = isinstance(test_data[0], list) and isinstance(test_data[1], list)

    if is_varied:
        print(f"Generating predictions with varied sensor locations for plotting ({title_prefix})...")
        x_branch_values_list, x_branch_locs_list, x_trunk_grid_tensor = test_data
        num_test_funcs = len(x_branch_values_list)
        # Convert lists to numpy arrays for easier indexing if needed later
        x_branch_values_np = np.array(x_branch_values_list)
        x_branch_locs_np = np.array(x_branch_locs_list)
    else:
        # Fixed sensor data format might need adjustment if used here,
        # assuming (x_branch_test_np, x_trunk_grid_tensor)
        print(f"Generating predictions with fixed sensor locations for plotting ({title_prefix})...")
        x_branch_test_np, x_trunk_grid_tensor = test_data
        num_test_funcs = x_branch_test_np.shape[0]
        # Get the default fixed sensor locations from the network's branch net for plotting
        # Ensure net is the CustomDeepONet instance
        default_sensor_locs_np = net.branch_net.default_x_sensors.squeeze(0).cpu().numpy()

    # Ensure grid is numpy
    grid_np = grid if isinstance(grid, np.ndarray) else grid.cpu().numpy()
    # Ensure grid_tensor is on the correct device
    device = next(net.parameters()).device
    if isinstance(x_trunk_grid_tensor, np.ndarray):
         x_trunk_grid_tensor = torch.from_numpy(x_trunk_grid_tensor).to(device)
    else:
         x_trunk_grid_tensor = x_trunk_grid_tensor.to(device)


    # --- Custom Prediction Loop ---
    y_pred_list = []
    net.eval() # Set the PyTorch model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        # Process in batches for potentially large test sets (e.g., batch size 64)
        test_batch_size = 64
        for i_start in range(0, num_test_funcs, test_batch_size):
            i_end = min(i_start + test_batch_size, num_test_funcs)
            current_batch_size = i_end - i_start

            # Prepare inputs for this batch
            trunk_input_tensor_batch = x_trunk_grid_tensor # Trunk input is fixed

            if is_varied:
                # Slice the numpy arrays. np.stack should now work correctly
                # as x_branch_values_np and x_branch_locs_np are float arrays.
                batch_vals_np = np.stack(x_branch_values_np[i_start:i_end])
                batch_locs_np = np.stack(x_branch_locs_np[i_start:i_end])
                # Convert batch to tensors
                branch_vals_tensor = torch.from_numpy(batch_vals_np).to(device) # This should work now
                branch_locs_tensor = torch.from_numpy(batch_locs_np).to(device)
                branch_input = (branch_vals_tensor, branch_locs_tensor) # Tuple for branch net
            else:
                # Slice the numpy array for the batch
                batch_vals_np = x_branch_test_np[i_start:i_end]
                # Convert batch to tensor
                branch_vals_tensor = torch.from_numpy(batch_vals_np).to(device)
                branch_input = branch_vals_tensor # Just the tensor

            # Create the tuple input for the CustomDeepONet forward method
            # Ensure trunk input is broadcast correctly if needed
            # If trunk_input_tensor_batch is (N_POINTS, 1) and branch output is (batch_size, p),
            # the einsum in CustomDeepONet handles the broadcasting.
            model_input = (branch_input, trunk_input_tensor_batch)

            # Get prediction from the PyTorch network
            prediction_tensor = net(model_input) # Shape (current_batch_size, n_points)
            y_pred_list.extend(prediction_tensor.cpu().numpy()) # Append batch results

    # --- End Custom Prediction Loop ---

    y_pred_np = np.array(y_pred_list) # Shape (num_test_funcs, n_points)

    # Ensure y_test is NumPy array
    y_test_np = np.array(true_values) # true_values should be list or array

    # --- Plotting Logic (remains largely the same) ---
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    plot_indices = np.random.choice(num_test_funcs, min(n_plots, num_test_funcs), replace=False)

    for i, func_idx in enumerate(plot_indices):
        ax = axs[i]
        true_coeffs = coefficients[func_idx]

        # Get sensor data for this specific function index
        if is_varied:
            # Need to access the potentially variable-sized arrays correctly
            sensor_locs_np_plot = x_branch_locs_np[func_idx].flatten()
            sensor_vals_np_plot = x_branch_values_np[func_idx].flatten()
            sensor_label = f"Input Sensors ({len(sensor_locs_np_plot)} varied points)"
        else:
            sensor_locs_np_plot = default_sensor_locs_np.flatten()
            sensor_vals_np_plot = x_branch_test_np[func_idx].flatten()
            sensor_label = f"Input Sensors ({len(sensor_locs_np_plot)} fixed points)"

        # Plot true function
        ax.plot(grid_np.flatten(), y_test_np[func_idx].flatten(), label="True Function", linewidth=2, color='blue')
        # Plot prediction
        ax.plot(grid_np.flatten(), y_pred_np[func_idx].flatten(), label="DeepONet Prediction", linestyle='--', linewidth=2, color='red')
        # Plot sensor locations
        ax.scatter(sensor_locs_np_plot, sensor_vals_np_plot, label=sensor_label, color='green', marker='o', s=50, zorder=5)

        title = f"$y = {true_coeffs[0]:.2f}x^2 + {true_coeffs[1]:.2f}x + {true_coeffs[2]:.2f}$"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if i == 0:
            ax.legend()

    # ... (rest of plotting logic: hide unused axes, set suptitle, savefig, close) ...
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.suptitle(f"{title_prefix}: DeepONet Predictions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(log_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Saved prediction plot to {plot_path}")
    plt.close(fig)

# --- Modify plot_loss_history to accept lists/arrays ---
def plot_loss_history(train_losses, test_losses, test_metrics, steps, log_dir):
    """Plots the training and testing loss/metric history from lists/arrays."""
    print("Plotting loss history...")
    # Ensure inputs are numpy arrays
    loss_train = np.array(train_losses)
    loss_test = np.array(test_losses)
    metric_test = np.array(test_metrics) # Assuming single metric still
    steps = np.array(steps)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss (MSE)', color=color)
    ax1.semilogy(steps, loss_train, color=color, linestyle='-', label='Train Loss')
    ax1.semilogy(steps, loss_test, color=color, linestyle='--', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean L2 Relative Error', color=color)
    ax2.semilogy(steps, metric_test, color=color, linestyle=':', label='Test Metric (L2 Rel Error)')
    ax2.tick_params(axis='y', labelcolor=color)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title('Training Loss and Test Metric History')
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    plot_path = os.path.join(log_dir, "loss_history.png")
    plt.savefig(plot_path)
    print(f"Saved loss history plot to {plot_path}")
    plt.close(fig)

# --- Modify plot_trunk_basis and plot_orthonormality_analysis ---
# These need to accept the 'net' instead of 'model'
def plot_trunk_basis(net, grid, log_dir, n_basis_to_plot=9):
    """Plots the first few learned basis functions from the trunk network."""
    print("Plotting trunk network basis functions...")
    device = next(net.parameters()).device
    grid_tensor = torch.from_numpy(grid).to(device) if isinstance(grid, np.ndarray) else grid.to(device)
    grid_np = grid if isinstance(grid, np.ndarray) else grid.cpu().numpy()
    # Access trunk_net directly from the CustomDeepONet instance
    trunk_net = net.trunk_net
    trunk_net.eval()
    with torch.no_grad():
        basis_values_tensor = trunk_net(grid_tensor)
    basis_values_np = basis_values_tensor.cpu().numpy()
    latent_dim = basis_values_np.shape[1]
    num_plots = min(n_basis_to_plot, latent_dim)
    if num_plots == 0:
        print("No basis functions to plot (latent_dim=0 or n_basis_to_plot=0).")
        return
    n_cols = 3
    n_rows = (num_plots + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    axs = axs.flatten()
    for i in range(num_plots):
        ax = axs[i]
        ax.plot(grid_np.flatten(), basis_values_np[:, i], linewidth=2)
        ax.set_title(f"Trunk Basis Function {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])
    plt.suptitle('Learned Trunk Network Basis Functions', fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    plot_path = os.path.join(log_dir, "trunk_basis_functions.png")
    plt.savefig(plot_path)
    print(f"Saved trunk basis functions plot to {plot_path}")
    plt.close(fig)


def plot_orthonormality_analysis(net, grid, log_dir):
    """
    Calculates and plots the Gram matrix of the trunk basis functions
    and computes the orthonormality error ||G - I||_F.
    """
    print("Analyzing trunk basis orthonormality...")
    # --- Get Basis Functions (using 'net') ---
    device = next(net.parameters()).device
    grid_tensor = torch.from_numpy(grid).to(device) if isinstance(grid, np.ndarray) else grid.to(device)
    grid_np = grid if isinstance(grid, np.ndarray) else grid.cpu().numpy()
    trunk_net = net.trunk_net # Access trunk_net directly
    trunk_net.eval()
    with torch.no_grad():
        basis_values_tensor = trunk_net(grid_tensor)
    basis_values_np = basis_values_tensor.cpu().numpy()
    latent_dim = basis_values_np.shape[1]
    n_points = basis_values_np.shape[0]
    if latent_dim == 0:
        print("No basis functions to analyze (latent_dim=0).")
        return
    gram_matrix = np.zeros((latent_dim, latent_dim))
    x_coords = grid_np.flatten()
    for i in range(latent_dim):
        for j in range(i, latent_dim):
            integrand = basis_values_np[:, i] * basis_values_np[:, j]
            integral_val = np.trapz(integrand, x_coords)
            gram_matrix[i, j] = integral_val
            gram_matrix[j, i] = integral_val
    identity_matrix = np.identity(latent_dim)
    orthonormality_error = np.linalg.norm(gram_matrix - identity_matrix, 'fro')
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(gram_matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Integral Value')
    ax.set_xticks(np.arange(latent_dim))
    ax.set_yticks(np.arange(latent_dim))
    ax.set_xticklabels(np.arange(1, latent_dim + 1))
    ax.set_yticklabels(np.arange(1, latent_dim + 1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    title = f'Trunk Basis Gram Matrix\nOrthonormality Error $||G - I||_F = {orthonormality_error:.4f}$'
    ax.set_title(title)
    fig.tight_layout()
    plot_path = os.path.join(log_dir, "orthonormality_analysis.png")
    plt.savefig(plot_path)
    print(f"Saved orthonormality analysis plot to {plot_path}")
    plt.close(fig)


def main():
    # Parameters
    # n_train_funcs = 1000 # Total funcs per epoch can be implicitly defined by steps * batch_size
    n_test_funcs = 200 # Number of functions for validation set
    learning_rate = 0.0001
    # epochs = 200000 # Define training length by steps instead
    train_steps = 300000
    batch_size = 64
    N_SENSORS_TRAIN = 30 # Number of sensors for TRAINING data (now varied per sample)
    N_SENSORS_VAL = 50   # Number of sensors for VALIDATION data (can be fixed or varied)
    val_freq = 1000      # How often to run validation (every N steps)

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --- End Determine Device ---

    # --- Create Log Directory ---
    base_log_dir = "deeponet_deepsets_pytorch_train_logs" # New log dir name
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs and plots will be saved in: {log_dir}")
    # --- End Log Directory Creation ---

    # --- Generate FIXED Sensor Locations for Branch Initialization (Optional but good practice) ---
    # The branch net still needs *some* default sensor shape info, even if not used for training calcs
    x_sensors_init_np = np.linspace(INPUT_RANGE[0], INPUT_RANGE[1], N_SENSORS_VAL).reshape(-1, 1).astype(np.float32)
    x_sensors_init_tensor = torch.from_numpy(x_sensors_init_np).to(device)
    # --- End Fixed Sensor Locations ---

    # --- Generate Grid ---
    grid = np.linspace(INPUT_RANGE[0], INPUT_RANGE[1], N_POINTS).reshape(-1, 1).astype(np.float32)
    grid_tensor = torch.from_numpy(grid).to(device)
    # --- End Grid ---

    # --- Generate Validation Data (Using Varied Sensors) ---
    # We'll use varied sensors for validation as well to get a true measure of generalization
    print(f"Generating validation data ({n_test_funcs} functions, {N_SENSORS_VAL} varied sensors each)...")
    val_branch_vals_list = []
    val_branch_locs_list = []
    val_y_true_list = []
    val_coeffs_list = []
    # Generate validation data sample by sample (or could do in one batch)
    for _ in range(n_test_funcs):
         # Use a modified generator that also returns coeffs
         a = np.random.uniform(A_RANGE[0], A_RANGE[1], size=(1, 1))
         b = np.random.uniform(B_RANGE[0], B_RANGE[1], size=(1, 1))
         c = np.random.uniform(C_RANGE[0], C_RANGE[1], size=(1, 1))
         coeffs_i = np.hstack((a, b, c)).astype(np.float32)
         x_sensors_i = np.sort(np.random.uniform(INPUT_RANGE[0], INPUT_RANGE[1], size=(N_SENSORS_VAL, 1))).astype(np.float32)
         y_sensors_i = quadratic_func(coeffs_i, x_sensors_i).flatten().astype(np.float32)
         y_grid_i = quadratic_func(coeffs_i, grid).flatten().astype(np.float32)

         val_branch_vals_list.append(y_sensors_i)
         val_branch_locs_list.append(x_sensors_i)
         val_y_true_list.append(y_grid_i)
         val_coeffs_list.append(coeffs_i.flatten()) # Store coeffs for plotting

    # Keep validation data as lists/numpy arrays for plot_results
    val_coeffs = np.array(val_coeffs_list)
    # Prepare tuple for plot_results (varied format)
    val_data_tuple_for_plot = (val_branch_vals_list, val_branch_locs_list, grid_tensor)
    # Convert validation ground truth to tensor for loss calculation
    val_y_true_tensor = torch.from_numpy(np.array(val_y_true_list)).to(device) # (n_test_funcs, n_points)
    print("Validation data generated.")
    # --- End Validation Data ---


    # --- Instantiate Network Structure ---
    latent_dim = 64
    phi_hidden_size = 128
    rho_hidden_size = 128
    trunk_layers = [1, 64, 64, latent_dim] # Trunk input dim is 1 (x coordinate)
    activation_fn = nn.ReLU

    # Create branch and trunk instances using imported classes
    branch_net = PyTorchDeepSetsBranch(
        x_sensors_tensor=x_sensors_init_tensor, # Use init tensor
        phi_hidden_size=phi_hidden_size,
        rho_hidden_size=rho_hidden_size,
        branch_output_dim=latent_dim,
        activation_fn=activation_fn
    ).to(device) # Move to device
    trunk_net = PyTorchTrunkNet(
        layer_sizes=trunk_layers,
        activation_fn=activation_fn
    ).to(device) # Move to device
    # Create the combined DeepONet structure using the imported class
    net = CustomDeepONet(branch_net, trunk_net).to(device) # Ensure the combined net is on device
    # --- End Network Structure Instantiation ---

    # --- Setup Optimizer and Loss ---
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # --- End Optimizer and Loss ---

    # --- Initialize Weights ---
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(init_weights) # Apply to the PyTorch model 'net'
    print("Applied Xavier initialization to custom DeepONet.")
    # --- End Initialization ---

    # --- Training Loop ---
    print(f"Starting PyTorch training for {train_steps} steps with batch size {batch_size}...")
    train_losses = []
    test_losses = []
    test_metrics = [] # L2 relative error
    steps_log = []

    net.train() # Set model to training mode

    for step in range(1, train_steps + 1):
        optimizer.zero_grad()

        # Generate a batch of training data with varied sensors
        batch_branch_vals_np, batch_branch_locs_np, batch_y_true_np = \
            generate_batch_with_varied_sensors(batch_size, N_SENSORS_TRAIN, grid)
        batch_branch_vals = torch.from_numpy(batch_branch_vals_np).to(device)
        batch_branch_locs = torch.from_numpy(batch_branch_locs_np).to(device)
        batch_y_true = torch.from_numpy(batch_y_true_np).to(device)

        # Prepare model input tuple
        branch_input = (batch_branch_vals, batch_branch_locs)
        # Ensure trunk input tensor is correctly shaped/broadcasted
        # grid_tensor is (N_POINTS, 1). CustomDeepONet handles the einsum.
        model_input = (branch_input, grid_tensor)

        # Forward pass
        predictions = net(model_input) # Shape (batch_size, n_points)

        # Calculate loss
        loss = criterion(predictions, batch_y_true)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Logging and Validation
        if step % val_freq == 0 or step == train_steps:
            net.eval() # Set model to evaluation mode for validation
            with torch.no_grad():
                # --- Calculate Training Loss (on last batch) ---
                train_loss_val = loss.item()
                train_losses.append(train_loss_val)

                # --- Calculate Validation Loss & Metric ---
                val_pred_list = []
                val_batch_size = 128 # Adjust as needed
                for i_start in range(0, n_test_funcs, val_batch_size):
                    i_end = min(i_start + val_batch_size, n_test_funcs)
                    # Stack validation data for batch processing
                    val_batch_vals_np = np.stack(val_branch_vals_list[i_start:i_end])
                    val_batch_locs_np = np.stack(val_branch_locs_list[i_start:i_end])
                    val_batch_vals = torch.from_numpy(val_batch_vals_np).to(device)
                    val_batch_locs = torch.from_numpy(val_batch_locs_np).to(device)

                    val_branch_input = (val_batch_vals, val_batch_locs)
                    val_model_input = (val_branch_input, grid_tensor) # grid_tensor is broadcast
                    val_pred_batch = net(val_model_input)
                    val_pred_list.append(val_pred_batch)

                val_predictions = torch.cat(val_pred_list, dim=0)
                test_loss_val = criterion(val_predictions, val_y_true_tensor).item()
                test_losses.append(test_loss_val)

                l2_error_norm = torch.linalg.norm(val_y_true_tensor - val_predictions, dim=1)
                l2_true_norm = torch.linalg.norm(val_y_true_tensor, dim=1)
                epsilon = 1e-8
                l2_rel_error = torch.mean(l2_error_norm / (l2_true_norm + epsilon)).item()
                test_metrics.append(l2_rel_error)

                steps_log.append(step)

                print(f"Step: {step}/{train_steps} | Train Loss: {train_loss_val:.4e} | Test Loss: {test_loss_val:.4e} | Test L2RelError: {l2_rel_error:.4e}")

            net.train() # Set model back to training mode

    print("\nTraining done.\n")
    # --- End Training Loop ---

    # --- Save the FINAL Model State ---
    final_model_path = os.path.join(log_dir, "final_model.pt")
    torch.save(net.state_dict(), final_model_path)
    print(f"Saved final model state to {final_model_path}")
    # --- End Saving Final Model ---

    # Ensure model is in eval mode for plotting using the final state
    net.eval()

    # --- Plotting ---
    # Plot results using the validation data (which used varied sensors)
    # This will now use the FINAL model state
    plot_results(net=net, # Pass the PyTorch net
                 test_data=val_data_tuple_for_plot, # Use the prepared tuple
                 true_values=val_y_true_list, # Pass the list/array of true values
                 coefficients=val_coeffs, # Pass the corresponding coefficients
                 grid=grid,
                 log_dir=log_dir,
                 plot_filename="predictions_varied_sensors_validation.png",
                 title_prefix="Varied Sensors Validation Set (Final Model)") # Updated title

    # Plot loss history
    plot_loss_history(train_losses, test_losses, test_metrics, steps_log, log_dir)

    # Plot trunk basis functions
    plot_trunk_basis(net, grid, log_dir) # Pass net

    # Plot orthonormality analysis
    plot_orthonormality_analysis(net, grid, log_dir) # Pass net

    print(f"\n--- Experiment Summary ---")
    print(f"Trained with PyTorch loop for {train_steps} steps.")
    print(f"Training data used {N_SENSORS_TRAIN} RANDOMLY chosen sensor locations per function.")
    print(f"Validation data used {N_SENSORS_VAL} RANDOMLY chosen sensor locations per function.")
    print(f"Saved FINAL model state.") # Updated summary message
    print(f"Results saved in: {log_dir}")
    print(f"--- End Summary ---")


if __name__ == "__main__":
    main()
