import torch
import torch.nn as nn
from FunctionEncoder import BaseDataset, BaseCallback # Assuming these are defined elsewhere
from tqdm import trange

# This implements a DeepOSet using PyTorch
class DeepOSet(torch.nn.Module):

    def __init__(self,
                 input_size_src,      # Dimensionality of sensor location x_i (e.g., 1 for 1D)
                 output_size_src,     # Dimensionality of sensor value u(x_i) (e.g., 1 for scalar)
                 input_size_tgt,      # Dimensionality of trunk input y (e.g., 1 for 1D)
                 output_size_tgt,     # Dimensionality of final output G(u)(y) (e.g., 1 for scalar)
                 # n_input_sensors is not strictly needed if handling variable sensors, but kept for consistency/potential use
                 p=32,                # Latent dimension for the branch/trunk cross product (Default)
                 phi_hidden_size=256, # Hidden layer size for the phi network (Default)
                 rho_hidden_size=256, # Hidden layer size for the rho network (Default)
                 trunk_hidden_size=256,# Hidden layer size for the trunk network (MLP) (Default)
                 n_trunk_layers=4,    # Number of layers in the trunk network (Default)
                 activation_fn=nn.ReLU, # Activation function
                 use_deeponet_bias=True, # Whether to use a bias term after the cross product
                 ):
        super().__init__()

        # set hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt

        self.p = p
        self.phi_hidden_size = phi_hidden_size
        self.rho_hidden_size = rho_hidden_size
        self.trunk_hidden_size = trunk_hidden_size
        self.n_trunk_layers = n_trunk_layers

        # --- Branch Network (Deep Sets) ---
        # Phi network: processes individual (location, value) pairs
        # Input dim: location_dim + value_dim
        phi_input_dim = input_size_src + output_size_src
        self.phi = nn.Sequential(
            nn.Linear(phi_input_dim, phi_hidden_size),
            activation_fn(),
            nn.Linear(phi_hidden_size, phi_hidden_size),
            activation_fn()
            # Output dim: phi_hidden_size
        )

        # Rho network: processes aggregated representation from phi
        # Input dim: phi_hidden_size (after aggregation)
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden_size, rho_hidden_size),
            activation_fn(),
            nn.Linear(rho_hidden_size, output_size_tgt * p) # Final branch output (before reshape)
            # Output dim: output_size_tgt * p
        )
        # --- End Branch Network ---

        # --- Trunk Network (MLP) ---
        # Maps y to t_1, ..., t_p (potentially multi-dimensional output_size_tgt)
        trunk_layers = []
        trunk_layers.append(nn.Linear(input_size_tgt, trunk_hidden_size))
        trunk_layers.append(activation_fn())
        for _ in range(n_trunk_layers - 2):
            trunk_layers.append(nn.Linear(trunk_hidden_size, trunk_hidden_size))
            trunk_layers.append(activation_fn())
        trunk_layers.append(nn.Linear(trunk_hidden_size, output_size_tgt * p))
        # Optional: Add Sigmoid like in DeepONet? Or keep linear for basis? Let's keep linear for now.
        # trunk_layers.append(torch.nn.Sigmoid())
        self.trunk = torch.nn.Sequential(*trunk_layers)
        # --- End Trunk Network ---

        # an optional bias, see equation 2 in the DeepONet paper.
        self.bias = torch.nn.Parameter(torch.randn(output_size_tgt) * 0.1) if use_deeponet_bias else None

        # create optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=5e-6)

        # holdovers from function encoder code, these do nothing
        self.method = "deepOSet"
        self.average_function = None

    def forward_branch(self, xs, us):
        """
        Forward pass for the Deep Sets Branch.
        Args:
            xs (torch.Tensor): Sensor locations, shape (batch_size, n_sensors, input_size_src)
            us (torch.Tensor): Sensor values, shape (batch_size, n_sensors, output_size_src)
        Returns:
            torch.Tensor: Branch output, shape (batch_size, p, output_size_tgt)
        """
        batch_size = xs.shape[0]
        n_sensors = xs.shape[1]

        # Concatenate location and value for each sensor
        # Shape: (batch_size, n_sensors, input_size_src + output_size_src)
        phi_input = torch.cat((xs, us), dim=2)

        # Reshape for phi's Linear layer: (batch_size * n_sensors, input_size_src + output_size_src)
        phi_input_reshaped = phi_input.view(batch_size * n_sensors, -1)

        # Apply phi to each (location, value) pair
        # Shape: (batch_size * n_sensors, phi_hidden_size)
        phi_output = self.phi(phi_input_reshaped)

        # Reshape back for aggregation
        # Shape: (batch_size, n_sensors, phi_hidden_size)
        phi_output_reshaped = phi_output.view(batch_size, n_sensors, self.phi_hidden_size)

        # Aggregate over the sensor dimension (dim=1) using mean pooling
        # Shape: (batch_size, phi_hidden_size)
        aggregated = torch.mean(phi_output_reshaped, dim=1)

        # Apply rho to the aggregated representation
        # Shape: (batch_size, output_size_tgt * p)
        rho_output = self.rho(aggregated)

        # Reshape rho output to match the desired structure for einsum
        # Shape: (batch_size, p, output_size_tgt)
        branch_out = rho_output.view(batch_size, self.p, self.output_size_tgt)

        return branch_out

    def forward_trunk(self, ys):
        """
        Forward pass for the Trunk Network.
        Args:
            ys (torch.Tensor): Trunk input locations, shape (batch_size, n_points, input_size_tgt)
        Returns:
            torch.Tensor: Trunk output, shape (batch_size, n_points, p, output_size_tgt)
        """
        batch_size = ys.shape[0]
        n_points = ys.shape[1]

        # Reshape for trunk's Linear layers if needed (MLP handles batch dim)
        # Input shape: (batch_size, n_points, input_size_tgt)
        # Output shape: (batch_size, n_points, output_size_tgt * p)
        trunk_out_flat = self.trunk(ys)

        # Reshape trunk output for einsum
        # Shape: (batch_size, n_points, p, output_size_tgt)
        trunk_out = trunk_out_flat.view(batch_size, n_points, self.p, self.output_size_tgt)
        return trunk_out

    def forward(self, xs, us, ys):
        """
        Full forward pass for DeepOSet.
        Args:
            xs (torch.Tensor): Sensor locations, shape (batch_size, n_sensors, input_size_src)
            us (torch.Tensor): Sensor values, shape (batch_size, n_sensors, output_size_src)
            ys (torch.Tensor): Trunk input locations, shape (batch_size, n_points, input_size_tgt)
        Returns:
            torch.Tensor: Predicted output G(u)(y), shape (batch_size, n_points, output_size_tgt)
        """
        # Get branch and trunk outputs
        b = self.forward_branch(xs, us) # Shape: (batch, p, out_tgt)
        t = self.forward_trunk(ys)      # Shape: (batch, n_points, p, out_tgt)

        # Combine using einsum (dot product over latent dimension p)
        # einsum: "bpz, bdpz -> bdz" (b=batch, d=n_points, p=latent_dim, z=out_tgt_dim)
        G_u_y = torch.einsum("bpz,bdpz->bdz", b, t)

        # optionally add bias
        if self.bias is not None:
            # Bias shape is (output_size_tgt), needs broadcasting to (batch, n_points, output_size_tgt)
            G_u_y = G_u_y + self.bias # Broadcasting handles the addition

        return G_u_y

    # This is the main training loop, kept consistent with the function encoder code.
    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback: BaseCallback = None):
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())


        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            # sample input data - dataset needs to provide xs, us, ys, G_u_ys
            # Ensure dataset.sample() returns locations (xs) and values (us) separately
            xs, us, ys, G_u_ys, _ = dataset.sample(device=device)


            # approximate functions, compute error
            estimated_G_u_ys = self.forward(xs, us, ys)
            prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)

            # add loss components (can add regularization later if needed)
            loss = prediction_loss

            # backprop with gradient clipping
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            # update progress bar
            if progress_bar:
                bar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4e} | Grad Norm: {norm:.2f}")

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())

    def _param_string(self):
        """ Returns a dictionary of hyperparameters for logging."""
        params = {}
        params["method"] = self.method
        params["input_size_src"] = self.input_size_src
        params["output_size_src"] = self.output_size_src
        params["input_size_tgt"] = self.input_size_tgt
        params["output_size_tgt"] = self.output_size_tgt
        params["p"] = self.p
        params["phi_hidden_size"] = self.phi_hidden_size
        params["rho_hidden_size"] = self.rho_hidden_size
        params["trunk_hidden_size"] = self.trunk_hidden_size
        params["n_trunk_layers"] = self.n_trunk_layers
        # params["activation_fn"] = self.phi[1].__class__.__name__ # Get activation class name
        params["use_deeponet_bias"] = self.bias is not None
        params = {k: str(v) for k, v in params.items()}
        return params
