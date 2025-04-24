import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import inspect

# --- Import the DeepOSets Branch ---
from deeposets_module import PyTorchDeepSetsBranch
# --- End Import ---

# --- Define PyTorch MLP Branch Locally (for Comparison) ---
class PyTorchMLPBranch(nn.Module):
    """
    Standard MLP for the branch network (NOT permutation invariant).
    Takes flattened sensor values as input. Defined locally in this script.
    """
    def __init__(self, n_sensors, hidden_size, branch_output_dim, activation_fn=nn.ReLU):
        """
        Initializes the MLP Branch network.

        Args:
            n_sensors (int): The number of sensor inputs.
            hidden_size (int): Hidden layer size for the MLP.
            branch_output_dim (int): Output dimension of the branch network (latent dimension p).
            activation_fn (torch.nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
        """
        super().__init__()
        self.n_sensors = n_sensors
        self.network = nn.Sequential(
            nn.Linear(n_sensors, hidden_size), # Input is flattened sensor values
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, branch_output_dim) # Final branch output
        )

    def forward(self, x):
        """
        Forward pass for the MLP Branch.

        Args:
            x (torch.Tensor): Input sensor values, shape (batch_size, n_sensors).

        Returns:
            torch.Tensor: Output of the branch network, shape (batch_size, branch_output_dim).
        """
        # Input validation
        if x.ndim != 2 or x.shape[1] != self.n_sensors:
             raise ValueError(f"MLP Branch expected input shape (batch_size, {self.n_sensors}), but got {x.shape}")
        return self.network(x)
# --- End Local PyTorch MLP Branch ---

# --- Debug Backend ---
print(f"PyTorch version: {torch.__version__}")
# --- End Debug ---

# Define ranges and parameters
A_RANGE = (-1.5, 1.5)
B_RANGE = (-1.5, 1.5)
C_RANGE = (-1.5, 1.5)
INPUT_RANGE = (-10, 10)
N_SENSORS = 50 # Number of sensor points

def quadratic_func(coeffs, x):
    """Calculates y = ax^2 + bx + c. Uses NumPy."""
    a, b, c = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
    if x.ndim == 1: x_t = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[1] == 1: x_t = x.T
    else: x_t = x
    return a * (x_t**2) + b * x_t + c

def get_data_for_pi_test(n_funcs, n_sensors):
    """
    Generates data for permutation invariance test.
    Creates original (x,y) pairs and permuted (x,y) pairs.
    """
    a_coeffs = np.random.uniform(A_RANGE[0], A_RANGE[1], size=(n_funcs, 1))
    b_coeffs = np.random.uniform(B_RANGE[0], B_RANGE[1], size=(n_funcs, 1))
    c_coeffs = np.random.uniform(C_RANGE[0], C_RANGE[1], size=(n_funcs, 1))
    coeffs = np.hstack((a_coeffs, b_coeffs, c_coeffs)).astype(np.float32)

    # Original sensor locations
    x_sensors_np = np.linspace(INPUT_RANGE[0], INPUT_RANGE[1], n_sensors).reshape(-1, 1).astype(np.float32)
    # Original sensor values
    y_sensors_np = quadratic_func(coeffs, x_sensors_np).astype(np.float32) # (n_funcs, n_sensors)

    # Create Permuted Data (Pairs)
    permutation_indices = np.random.permutation(n_sensors)
    x_sensors_permuted_np = x_sensors_np[permutation_indices]
    y_sensors_permuted_np = y_sensors_np[:, permutation_indices]

    print(f"Generated data for {n_funcs} function(s).")
    print(f"Original y_sensors shape: {y_sensors_np.shape}, x_sensors shape: {x_sensors_np.shape}")
    print(f"Permuted y_sensors shape: {y_sensors_permuted_np.shape}, x_sensors shape: {x_sensors_permuted_np.shape}")

    return x_sensors_np, y_sensors_np, x_sensors_permuted_np, y_sensors_permuted_np

# --- MODIFIED: run_pi_test returns results ---
def run_pi_test(branch_net,
                y_original_tensor, x_original_tensor_batch,
                y_permuted_tensor, x_permuted_tensor_batch,
                network_type, aggregation_mode=None):
    """
    Runs the PI test and returns coefficients and max difference.
    """
    branch_net.eval()

    # Prepare input based on network type
    if network_type == 'deepsets':
        input_original = (y_original_tensor, x_original_tensor_batch)
        input_permuted = (y_permuted_tensor, x_permuted_tensor_batch)
        mode_str = f" (Aggregation: {aggregation_mode.upper()})" if aggregation_mode else ""
        test_desc = f"DeepSets ({aggregation_mode.upper()})"
        pi_desc = "permutation of (value, location) pairs"
    elif network_type == 'mlp':
        # MLP branch expects only flattened sensor values: (batch_size, n_sensors)
        # NOTE: We pass the *original* ordered y values for MLP test consistency
        input_original = y_original_tensor
        # For the 'permuted' MLP input, we use the y values that were permuted
        # relative to the original x locations. This is the standard way to show
        # MLP's lack of invariance to value order.
        input_permuted = y_permuted_tensor
        mode_str = " (Standard MLP)"
        test_desc = "Standard MLP"
        pi_desc = "permutation of values relative to fixed positions"
    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    # Perform inference
    with torch.no_grad():
        coeffs_original = branch_net(input_original)
        coeffs_permuted = branch_net(input_permuted)

    # Compare Outputs
    print(f"\n--- Permutation Invariance Test Results{mode_str} ---")
    print(f"Testing invariance to {pi_desc} for {test_desc}")
    print(f"Output shape: {coeffs_original.shape}")

    # Calculate the difference
    difference = torch.abs(coeffs_original - coeffs_permuted)
    max_diff = torch.max(difference).item()
    mean_diff = torch.mean(difference).item()
    sum_diff = torch.sum(difference).item()

    # Print results
    print(f"\nOriginal Coefficients (first 5): {coeffs_original.cpu().numpy().flatten()[:5]}")
    print(f"Permuted Coefficients (first 5): {coeffs_permuted.cpu().numpy().flatten()[:5]}")
    print(f"\nMaximum Absolute Difference: {max_diff:.6e}")
    print(f"Mean Absolute Difference:    {mean_diff:.6e}")
    print(f"Sum Absolute Difference:     {sum_diff:.6e}")

    # Check tolerance
    tolerance = 1e-4 # Adjusted tolerance
    is_invariant = max_diff < tolerance

    if network_type == 'deepsets':
        if is_invariant:
            print(f"\n✅ Test Passed: Difference below tolerance ({tolerance:.1e}). {test_desc} appears correctly permutation invariant to (value, location) pairs.")
        else:
            print(f"\n❌ Test Failed: Difference ({max_diff:.6e}) exceeds tolerance ({tolerance:.1e}). {test_desc} FAILED PI test for (value, location) pairs.")
            print(f"   Check implementation, especially torch.{aggregation_mode} and phi network.")
    elif network_type == 'mlp':
        if is_invariant:
            print(f"\n⚠️ Test Warning: Difference below tolerance ({tolerance:.1e}). {test_desc} appears permutation invariant (UNEXPECTED).")
        else:
            # Expected outcome for MLP
            print(f"\n✅ Test Passed (Expected Failure): Difference ({max_diff:.6e}) exceeds tolerance ({tolerance:.1e}). {test_desc} is NOT permutation invariant (as expected).")

    print("-" * 60) # Separator

    # Return results for plotting
    return coeffs_original.cpu().numpy().flatten(), coeffs_permuted.cpu().numpy().flatten(), max_diff, test_desc
# --- END MODIFICATION ---

# --- NEW Plotting Function ---
def plot_pi_results(results_list, filename="pi_test_scatter.png"):
    """
    Generates scatter plots comparing original vs permuted coefficients.

    Args:
        results_list (list): A list of tuples, where each tuple contains
                             (coeffs_original, coeffs_permuted, max_diff, test_desc)
                             for one test run.
        filename (str): The name of the file to save the plot.
    """
    num_plots = len(results_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), squeeze=False)
    axes = axes.flatten() # Ensure axes is always iterable

    for i, (coeffs_orig, coeffs_perm, max_diff, test_desc) in enumerate(results_list):
        ax = axes[i]
        # Scatter plot
        ax.scatter(coeffs_orig, coeffs_perm, alpha=0.6, label=f'Max Diff: {max_diff:.2e}', s=10)

        # Add y=x line for reference
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Original Coefficients")
        ax.set_ylabel("Permuted Coefficients")
        ax.set_title(f"Permutation Invariance: {test_desc}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nSaved permutation invariance scatter plot to: {filename}")
    plt.close(fig) # Close the figure to free memory
# --- END NEW Plotting Function ---

def main():
    # Parameters
    n_funcs_to_test = 1
    latent_dim = 64
    phi_hidden_size = 128
    rho_hidden_size = 128
    mlp_hidden_size = 128 # Hidden size for the local MLP branch
    activation_fn = nn.ReLU

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --- End Determine Device ---

    print("Generating data for permutation invariance test...")
    x_sensors_np, y_sensors_np, x_sensors_permuted_np, y_sensors_permuted_np = get_data_for_pi_test(
        n_funcs_to_test, N_SENSORS
    )

    # Convert numpy arrays to PyTorch tensors
    x_sensors_tensor = torch.from_numpy(x_sensors_np).to(device)
    x_sensors_permuted_tensor = torch.from_numpy(x_sensors_permuted_np).to(device)
    x_sensors_tensor_batch = x_sensors_tensor.unsqueeze(0)
    x_sensors_permuted_tensor_batch = x_sensors_permuted_tensor.unsqueeze(0)
    y_sensors_tensor = torch.from_numpy(y_sensors_np).to(device)
    y_sensors_permuted_tensor = torch.from_numpy(y_sensors_permuted_np).to(device)

    # ---------------------------------------------------------------
    # Detect the correct aggregation keyword once and print it
    # ---------------------------------------------------------------
    sig_params = inspect.signature(PyTorchDeepSetsBranch.__init__).parameters
    if "aggregation_mode" in sig_params:
        AGG_KW = "aggregation_mode"
    elif "aggregation_type" in sig_params:
        AGG_KW = "aggregation_type"
    elif "aggregation" in sig_params:
        AGG_KW = "aggregation"
    else:
        AGG_KW = None
    print(f"Detected PyTorchDeepSetsBranch aggregation keyword: {AGG_KW}")

    # --- Instantiate DeepSets Branch (MEAN Aggregation) ---
    print("\nInstantiating DeepSets Branch with MEAN aggregation...")
    branch_ds_mean = PyTorchDeepSetsBranch(
        x_sensors_tensor=x_sensors_tensor,
        phi_hidden_size=phi_hidden_size,
        rho_hidden_size=rho_hidden_size,
        branch_output_dim=latent_dim,
        activation_fn=activation_fn,
        **({AGG_KW: "mean"} if AGG_KW else {})
    ).to(device)
    print("DeepSets Branch (MEAN) Instantiated.")

    # --- Instantiate DeepSets Branch (SUM Aggregation) ---
    print("\nInstantiating DeepSets Branch with SUM aggregation...")
    branch_ds_sum = PyTorchDeepSetsBranch(
        x_sensors_tensor=x_sensors_tensor,
        phi_hidden_size=phi_hidden_size,
        rho_hidden_size=rho_hidden_size,
        branch_output_dim=latent_dim,
        activation_fn=activation_fn,
        **({AGG_KW: "sum"} if AGG_KW else {})
    ).to(device)
    print("DeepSets Branch (SUM) Instantiated.")

    # --- Instantiate DeepSets Branch (ATTENTION Aggregation) ---
    print("\nInstantiating DeepSets Branch with ATTENTION aggregation...")
    branch_ds_attention = PyTorchDeepSetsBranch(
        x_sensors_tensor=x_sensors_tensor,
        phi_hidden_size=phi_hidden_size,
        rho_hidden_size=rho_hidden_size,
        branch_output_dim=latent_dim,
        activation_fn=activation_fn,
        **({AGG_KW: "attention"} if AGG_KW else {})
    ).to(device)
    print("DeepSets Branch (ATTENTION) Instantiated.")

    # --- Instantiate Standard MLP Branch (Locally Defined) ---
    print("\nInstantiating Standard MLP Branch (Locally Defined)...")
    branch_mlp = PyTorchMLPBranch(
        n_sensors=N_SENSORS,
        hidden_size=mlp_hidden_size,
        branch_output_dim=latent_dim,
        activation_fn=activation_fn
    ).to(device)
    print("Standard MLP Branch Instantiated.")

    # --- Run Tests and Collect Results ---
    results_list = []

    # Test DeepSets MEAN
    coeffs_orig, coeffs_perm, max_diff, desc = run_pi_test(
        branch_ds_mean,
        y_sensors_tensor, x_sensors_tensor_batch,
        y_sensors_permuted_tensor, x_sensors_permuted_tensor_batch,
        network_type='deepsets', aggregation_mode='mean'
    )
    results_list.append((coeffs_orig, coeffs_perm, max_diff, desc))

    # Test DeepSets SUM
    coeffs_orig, coeffs_perm, max_diff, desc = run_pi_test(
        branch_ds_sum,
        y_sensors_tensor, x_sensors_tensor_batch,
        y_sensors_permuted_tensor, x_sensors_permuted_tensor_batch,
        network_type='deepsets', aggregation_mode='sum'
    )
    results_list.append((coeffs_orig, coeffs_perm, max_diff, desc))

    # Test DeepSets ATTENTION -----------------------------------------------
    coeffs_orig, coeffs_perm, max_diff, desc = run_pi_test(
        branch_ds_attention,
        y_sensors_tensor, x_sensors_tensor_batch,
        y_sensors_permuted_tensor, x_sensors_permuted_tensor_batch,
        network_type='deepsets', aggregation_mode='attention'
    )
    results_list.append((coeffs_orig, coeffs_perm, max_diff, desc))

    # Test Standard MLP
    coeffs_orig, coeffs_perm, max_diff, desc = run_pi_test(
        branch_mlp,
        y_sensors_tensor, None,
        y_sensors_permuted_tensor, None,
        network_type='mlp'
    )
    results_list.append((coeffs_orig, coeffs_perm, max_diff, desc))
    # --- End Tests ---

    # --- Generate Plot ---
    plot_pi_results(results_list, filename="permutation_invariance_comparison.png")
    # --- End Plot ---

if __name__ == "__main__":
    main()
