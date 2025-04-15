import deepxde as dde
import os # Make sure os is imported early
import numpy as np
import matplotlib.pyplot as plt
# Import PyTorch
import torch
import torch.nn as nn
from datetime import datetime # Import datetime for timestamp

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

def train_deeponet(model, lr, epochs, log_dir):
    """Compiles and trains the DeepONet model."""
    decay = ("inverse time", epochs // 10, 0.8)
    model.compile("adam", lr=lr, metrics=["mean l2 relative error"], decay=decay)

    print(f"Starting training for {epochs} epochs with learning rate {lr}...")
    # Train without the callback
    losshistory, train_state = model.train(
        epochs=epochs,
        batch_size=None
    )
    print("\nTraining done.\n")

    # Save the final model state
    save_path = os.path.join(log_dir, "model")
    model.save(save_path, protocol="pytorch")
    print(f"Final model state saved to {log_dir} with base name 'model'")

    return losshistory, train_state

def plot_results(model, x_test, y_test, test_coeffs, grid, log_dir, n_plots=9):
    """Plots predictions vs true values for a subset of test functions."""
    print("Generating predictions for plotting...")
    # x_test is (x_branch_test, grid) - DeepXDE handles backend conversion
    y_pred = model.predict(x_test)

    # Ensure grid, y_test, y_pred are NumPy arrays for plotting
    grid_np = grid if isinstance(grid, np.ndarray) else grid.numpy()
    y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.numpy()

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()

    num_test_funcs = y_test_np.shape[0]
    plot_indices = np.random.choice(num_test_funcs, min(n_plots, num_test_funcs), replace=False)

    for i, func_idx in enumerate(plot_indices):
        ax = axs[i]
        true_coeffs = test_coeffs[func_idx] # Already numpy
        ax.plot(grid_np.flatten(), y_test_np[func_idx].flatten(), label="True", linewidth=2)
        ax.plot(grid_np.flatten(), y_pred_np[func_idx].flatten(), label="Prediction", linestyle='--', linewidth=2)

        title = f"$y = {true_coeffs[0]:.2f}x^2 + {true_coeffs[1]:.2f}x + {true_coeffs[2]:.2f}$"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plot_path = os.path.join(log_dir, "quadratic_deeponet_deepsets_predictions.png")
    plt.savefig(plot_path)
    print(f"Saved prediction plot to {plot_path}")

def plot_loss_history(losshistory, log_dir):
    """Plots the training and testing loss history."""
    print("Plotting loss history...")
    loss_train = np.array(losshistory.loss_train).sum(axis=1)
    loss_test = np.array(losshistory.loss_test).sum(axis=1)
    metric_test = np.array(losshistory.metrics_test).sum(axis=1) # Assuming single metric

    # Use steps provided by losshistory
    steps = losshistory.steps

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and test loss on primary y-axis (log scale)
    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss (MSE)', color=color)
    ax1.semilogy(steps, loss_train, color=color, linestyle='-', label='Train Loss')
    ax1.semilogy(steps, loss_test, color=color, linestyle='--', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    # Instantiate a second axes that shares the same x-axis for the metric
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean L2 Relative Error', color=color)  # we already handled the x-label with ax1
    ax2.semilogy(steps, metric_test, color=color, linestyle=':', label='Test Metric (L2 Rel Error)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Training Loss and Test Metric History')

    # Adjust layout
    fig.tight_layout() # Apply tight layout first
    plt.subplots_adjust(top=0.92) # Then, slightly adjust the top margin to ensure title fits

    plot_path = os.path.join(log_dir, "loss_history.png")
    plt.savefig(plot_path)
    print(f"Saved loss history plot to {plot_path}")
    plt.close(fig) # Close the figure to free memory

def plot_trunk_basis(model, grid, log_dir, n_basis_to_plot=9):
    """Plots the first few learned basis functions from the trunk network."""
    print("Plotting trunk network basis functions...")

    # Determine device model is on
    # Access the underlying PyTorch model via model.net
    pytorch_model = model.net
    device = next(pytorch_model.parameters()).device
    # Ensure grid is a tensor on the correct device
    grid_tensor = torch.from_numpy(grid).to(device) if isinstance(grid, np.ndarray) else grid.to(device)
    grid_np = grid if isinstance(grid, np.ndarray) else grid.cpu().numpy() # For plotting x-axis

    # Get the trunk network and put it in evaluation mode
    # Access trunk_net from the CustomDeepONet instance
    trunk_net = pytorch_model.trunk_net
    trunk_net.eval()

    # Get basis function values (output of trunk net)
    with torch.no_grad(): # Disable gradient calculation for inference
        basis_values_tensor = trunk_net(grid_tensor)

    # Move to CPU and convert to numpy
    basis_values_np = basis_values_tensor.cpu().numpy() # Shape: (n_points, latent_dim)
    latent_dim = basis_values_np.shape[1]

    # Determine how many bases to actually plot
    num_plots = min(n_basis_to_plot, latent_dim)
    if num_plots == 0:
        print("No basis functions to plot (latent_dim=0 or n_basis_to_plot=0).")
        return

    # Setup subplots (e.g., 3x3 grid)
    n_cols = 3
    n_rows = (num_plots + n_cols - 1) // n_cols # Calculate rows needed
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    axs = axs.flatten()

    for i in range(num_plots):
        ax = axs[i]
        ax.plot(grid_np.flatten(), basis_values_np[:, i], linewidth=2)
        ax.set_title(f"Trunk Basis Function {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide unused subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle('Learned Trunk Network Basis Functions', fontsize=16)
    fig.tight_layout()
    # Adjust layout slightly to prevent suptitle overlap
    plt.subplots_adjust(top=0.92)

    plot_path = os.path.join(log_dir, "trunk_basis_functions.png")
    plt.savefig(plot_path)
    print(f"Saved trunk basis functions plot to {plot_path}")
    plt.close(fig) # Close the figure

def plot_orthonormality_analysis(model, grid, log_dir):
    """
    Calculates and plots the Gram matrix of the trunk basis functions
    and computes the orthonormality error ||G - I||_F.
    """
    print("Analyzing trunk basis orthonormality...")

    # --- Get Basis Functions (same as in plot_trunk_basis) ---
    pytorch_model = model.net # Access the underlying PyTorch model
    device = next(pytorch_model.parameters()).device
    grid_tensor = torch.from_numpy(grid).to(device) if isinstance(grid, np.ndarray) else grid.to(device)
    grid_np = grid if isinstance(grid, np.ndarray) else grid.cpu().numpy()

    trunk_net = pytorch_model.trunk_net # Access trunk_net from CustomDeepONet
    trunk_net.eval()
    with torch.no_grad():
        basis_values_tensor = trunk_net(grid_tensor)
    basis_values_np = basis_values_tensor.cpu().numpy() # Shape: (n_points, latent_dim)
    latent_dim = basis_values_np.shape[1]
    n_points = basis_values_np.shape[0]

    if latent_dim == 0:
        print("No basis functions to analyze (latent_dim=0).")
        return
    # --- End Get Basis Functions ---

    # --- Calculate Gram Matrix using Numerical Integration (Trapezoidal Rule) ---
    gram_matrix = np.zeros((latent_dim, latent_dim))
    # Use np.trapz for integration. Requires y-values then x-values.
    # Ensure grid_np is flattened for x-values in trapz
    x_coords = grid_np.flatten()

    for i in range(latent_dim):
        for j in range(i, latent_dim): # Compute only upper triangle + diagonal
            # Product of basis_i and basis_j at each grid point
            integrand = basis_values_np[:, i] * basis_values_np[:, j]
            # Integrate the product over the grid
            integral_val = np.trapz(integrand, x_coords)
            gram_matrix[i, j] = integral_val
            gram_matrix[j, i] = integral_val # Symmetric matrix

    # --- Calculate Orthonormality Error ---
    identity_matrix = np.identity(latent_dim)
    orthonormality_error = np.linalg.norm(gram_matrix - identity_matrix, 'fro')

    # --- Plot Gram Matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(gram_matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Integral Value')

    ax.set_xticks(np.arange(latent_dim))
    ax.set_yticks(np.arange(latent_dim))
    ax.set_xticklabels(np.arange(1, latent_dim + 1))
    ax.set_yticklabels(np.arange(1, latent_dim + 1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(latent_dim):
    #     for j in range(latent_dim):
    #         text = ax.text(j, i, f"{gram_matrix[i, j]:.2f}",
    #                        ha="center", va="center", color="w" if np.abs(gram_matrix[i, j]) > 0.5 else "black")


    title = f'Trunk Basis Gram Matrix\nOrthonormality Error $||G - I||_F = {orthonormality_error:.4f}$'
    ax.set_title(title)
    fig.tight_layout()
    # Adjust layout slightly if needed (might not be necessary here)
    # plt.subplots_adjust(top=0.90)

    plot_path = os.path.join(log_dir, "orthonormality_analysis.png")
    plt.savefig(plot_path)
    print(f"Saved orthonormality analysis plot to {plot_path}")
    plt.close(fig) # Close the figure

def main():
    # Parameters
    n_train_funcs = 1000
    n_test_funcs = 200
    learning_rate = 0.0001
    epochs = 100000 # Reduced for faster testing, increase as needed
    N_SENSORS = 50 # Set the number of sensors here

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --- End Determine Device ---

    # --- Create Log Directory ---
    base_log_dir = "deeponet_deepsets_pos_logs" # New log dir name
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs and plots will be saved in: {log_dir}")
    # --- End Log Directory Creation ---

    # --- Generate Sensor Locations ONCE ---
    # Shape: (N_SENSORS, 1)
    x_sensors_np = np.linspace(INPUT_RANGE[0], INPUT_RANGE[1], N_SENSORS).reshape(-1, 1).astype(np.float32)
    # Convert to PyTorch tensor ON THE CORRECT DEVICE
    x_sensors_tensor = torch.from_numpy(x_sensors_np).to(device)
    # --- End Sensor Locations ---

    print("Generating data...")
    # Pass numpy version of x_sensors to get_data (get_data uses numpy)
    x_train, y_train, x_test, y_test, test_coeffs, grid = get_data(
        n_train_funcs, n_test_funcs, N_POINTS, x_sensors_np
    )
    print(f"Branch input shape (y_sensors): {x_train[0].shape}")
    print(f"Trunk input shape (grid): {x_train[1].shape}")
    print(f"Output shape (y_data): {y_train.shape}")

    # --- Instantiate Network Structure ---
    # Define dimensions - UPDATED PARAMETERS
    latent_dim = 32  # Keep the same latent dimension
    phi_hidden_size = 128
    rho_hidden_size = 128
    # Updated trunk network: deeper and wider
    trunk_layers = [1, 128, 128, 128, latent_dim]  # Deeper and wider trunk network
    # Try a different activation function for potentially smoother approximations
    activation_fn = nn.SiLU  # Swish/SiLU activation instead of ReLU

    # Create branch and trunk instances using imported classes
    branch_net = PyTorchDeepSetsBranch(
        x_sensors_tensor=x_sensors_tensor, # Pass the tensor with default locations
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

    # Setup data object
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    # Create model
    model = dde.Model(data, net)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(init_weights) # Apply to the PyTorch model 'net'
    print("Applied Xavier initialization to custom DeepONet.")

    # Train the model and get loss history
    losshistory, train_state = train_deeponet(model, learning_rate, epochs, log_dir)

    # Plot results
    plot_results(model, x_test, y_test, test_coeffs, grid, log_dir)

    # Plot loss history
    plot_loss_history(losshistory, log_dir)

    # Plot trunk basis functions (final state)
    plot_trunk_basis(model, grid, log_dir)

    # Plot orthonormality analysis (final state)
    plot_orthonormality_analysis(model, grid, log_dir)


if __name__ == "__main__":
    main()
