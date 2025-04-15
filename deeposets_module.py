import torch
import torch.nn as nn

# --- Define PyTorch Deep Sets Model for Branch Net ---
class PyTorchDeepSetsBranch(nn.Module):
    """
    PyTorch implementation of Deep Sets for the branch network.
    PHI network now takes (value, position) as input.
    Can optionally accept sensor locations during forward pass.
    """
    def __init__(self, x_sensors_tensor, phi_hidden_size, rho_hidden_size, branch_output_dim, activation_fn=nn.ReLU):
        """
        Initializes the Deep Sets Branch network.

        Args:
            x_sensors_tensor (torch.Tensor): A tensor representing default sensor locations,
                                             shape (n_sensors, 1). Used if locations are not
                                             provided during the forward pass.
            phi_hidden_size (int): Hidden layer size for the phi network.
            rho_hidden_size (int): Hidden layer size for the rho network.
            branch_output_dim (int): Output dimension of the branch network (latent dimension p).
            activation_fn (torch.nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
        """
        super().__init__()
        # x_sensors_tensor should be shape (n_sensors, 1) - Used as default/during training
        n_sensors = x_sensors_tensor.shape[0]
        self.n_sensors = n_sensors # Default number of sensors

        # Register default x_sensors as a buffer (non-parameter tensor)
        # Reshape to (1, n_sensors, 1) for easier broadcasting during training
        self.register_buffer("default_x_sensors", x_sensors_tensor.view(1, n_sensors, 1))

        # Phi network: Input dimension is now 2 (value + position)
        self.phi = nn.Sequential(
            nn.Linear(2, phi_hidden_size), # Input is (value, position) pair
            activation_fn(),
            nn.Linear(phi_hidden_size, phi_hidden_size),
            activation_fn()
        )

        # Rho network (processes the aggregated representation)
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden_size, rho_hidden_size), # Input is aggregated phi output
            activation_fn(),
            nn.Linear(rho_hidden_size, branch_output_dim) # Final branch output
        )

    def forward(self, x):
        """
        Forward pass for the Deep Sets Branch.

        Args:
            x (torch.Tensor or tuple): Input data. Can be:
                1. Tensor of sensor values (batch_size, n_sensors) -> Uses default locations.
                2. Tuple (sensor_values, sensor_locs):
                   - sensor_values: Tensor (batch_size, n_sensors_batch)
                   - sensor_locs: Tensor (batch_size, n_sensors_batch, 1)

        Returns:
            torch.Tensor: Output of the branch network, shape (batch_size, branch_output_dim).
        """
        if isinstance(x, tuple):
            x_values, x_locs = x
            # x_locs shape: (batch_size, n_sensors_batch, 1)
            # x_values shape: (batch_size, n_sensors_batch)
            batch_size = x_values.shape[0]
            n_sensors_batch = x_values.shape[1] # Sensors can vary per batch if needed
            x_sensors_batch = x_locs # Use provided locations
            # Prepare sensor values: shape (batch_size, n_sensors_batch, 1)
            x_values = x_values.unsqueeze(-1)
        else:
            # Assume x contains only sensor values, use default locations
            x_values = x # Shape: (batch_size, self.n_sensors)
            batch_size = x_values.shape[0]
            n_sensors_batch = self.n_sensors
            # Prepare sensor values: shape (batch_size, n_sensors_batch, 1)
            x_values = x_values.unsqueeze(-1)
            # Prepare sensor locations: shape (batch_size, n_sensors_batch, 1)
            # self.default_x_sensors has shape (1, n_sensors, 1)
            x_sensors_batch = self.default_x_sensors.expand(batch_size, -1, -1)


        # Concatenate value and position along the feature dimension
        # Shape: (batch_size, n_sensors_batch, 2)
        phi_input = torch.cat((x_values, x_sensors_batch), dim=2)

        # Reshape for phi's Linear layer: (batch_size * n_sensors_batch, 2)
        phi_input_reshaped = phi_input.view(batch_size * n_sensors_batch, 2)

        # Apply phi to each (value, position) pair
        phi_output = self.phi(phi_input_reshaped) # Shape: (batch_size * n_sensors_batch, phi_hidden_size)

        # Reshape back and aggregate (mean pooling)
        # Shape: (batch_size, n_sensors_batch, phi_hidden_size)
        phi_output_reshaped = phi_output.view(batch_size, n_sensors_batch, -1)
        # Aggregate over the sensor dimension (dim=1)
        aggregated = torch.mean(phi_output_reshaped, dim=1) # Shape: (batch_size, phi_hidden_size)

        # Apply rho to the aggregated representation
        rho_output = self.rho(aggregated) # Shape: (batch_size, branch_output_dim)
        return rho_output
# --- End PyTorch Deep Sets Model ---

# --- Define PyTorch MLP for Trunk Net ---
class PyTorchTrunkNet(nn.Module):
    """Simple PyTorch MLP for the trunk network."""
    def __init__(self, layer_sizes, activation_fn=nn.ReLU):
        """
        Initializes the Trunk MLP network.

        Args:
            layer_sizes (list): List of integers defining the layer sizes,
                                e.g., [input_dim, hidden1, hidden2, output_dim].
            activation_fn (torch.nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1])) # No activation on final layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the Trunk MLP.

        Args:
            x (torch.Tensor): Input tensor, typically shape (n_points, input_dim) or (batch * n_points, input_dim).

        Returns:
            torch.Tensor: Output of the trunk network, shape (n_points, output_dim) or (batch * n_points, output_dim).
        """
        return self.network(x)
# --- End PyTorch Trunk Net ---

# --- Define Custom DeepONet Module ---
class CustomDeepONet(nn.Module):
    """
    Manually implements the DeepONet structure using separate branch/trunk modules.
    Combines a branch network (like Deep Sets) and a trunk network (MLP).
    """
    def __init__(self, branch_module, trunk_module):
        """
        Initializes the Custom DeepONet.

        Args:
            branch_module (torch.nn.Module): An initialized branch network module (e.g., PyTorchDeepSetsBranch).
            trunk_module (torch.nn.Module): An initialized trunk network module (e.g., PyTorchTrunkNet).
        """
        super().__init__()
        self.branch_net = branch_module
        self.trunk_net = trunk_module
        # Add the expected 'regularizer' attribute for compatibility with some frameworks (like DeepXDE)
        self.regularizer = None
        # Optional: Add a learnable bias term if needed (common in DeepONets)
        # self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass for the Custom DeepONet.

        Args:
            x (tuple): A tuple containing the inputs for the branch and trunk networks:
                       (x_branch_data, x_trunk)
                       - x_branch_data: Input for the branch network (Tensor or Tuple, see branch_net docs).
                       - x_trunk: Input for the trunk network, shape (n_points, trunk_input_dim)
                                  or potentially batched (batch_size, n_points, trunk_input_dim).

        Returns:
            torch.Tensor: The combined output of the DeepONet, typically shape (batch_size, n_points).
        """
        # x is expected to be a tuple: (x_branch_data, x_trunk)
        x_branch_data, x_trunk = x

        # Process branch input - branch_net handles if x_branch_data is tuple or tensor
        # branch_output shape: (batch_size, p)
        branch_output = self.branch_net(x_branch_data)

        # Process trunk input
        # trunk_output shape: (n_points, p) or (batch_size * n_points, p)
        # Need to handle potential batching of trunk input
        original_trunk_shape = x_trunk.shape
        if x_trunk.ndim == 3: # (batch_size, n_points, trunk_input_dim)
            n_points = x_trunk.shape[1]
            trunk_input_dim = x_trunk.shape[2]
            x_trunk_reshaped = x_trunk.view(-1, trunk_input_dim) # (batch_size * n_points, trunk_input_dim)
            trunk_output = self.trunk_net(x_trunk_reshaped) # (batch_size * n_points, p)
            # Reshape trunk output back to match batch structure: (batch_size, n_points, p)
            trunk_output = trunk_output.view(original_trunk_shape[0], n_points, -1)
        elif x_trunk.ndim == 2: # Assume (n_points, trunk_input_dim)
            n_points = x_trunk.shape[0]
            trunk_output = self.trunk_net(x_trunk) # (n_points, p)
        else:
            raise ValueError(f"Unexpected trunk input dimension: {x_trunk.ndim}. Expected 2 or 3.")


        # Combine using dot product (or einsum for clarity)
        if trunk_output.ndim == 3: # Batch dimension present in trunk output
            # einsum: "bi, bni -> bn" (b=batch, i=latent_dim, n=n_points)
            y = torch.einsum("bi, bni -> bn", branch_output, trunk_output)
        else: # No batch dimension in trunk output, broadcast branch output
             # einsum: "bi, ni -> bn" (b=batch, i=latent_dim, n=n_points)
            y = torch.einsum("bi, ni -> bn", branch_output, trunk_output)


        # Add bias if included
        # y = y + self.b

        return y
# --- End Custom DeepONet Module --- 