import torch
import torch.nn as nn
from FunctionEncoder import BaseDataset, BaseCallback # Keep BaseDataset/Callback if used
from tqdm import trange
from torch.optim.lr_scheduler import _LRScheduler # Import base class for type hinting if needed

# This implements a DeepOSet using PyTorch
class DeepOSet(torch.nn.Module):

    def __init__(self,
                 input_size_src,      # Dimensionality of sensor location x_i (e.g., 1 for 1D)
                 output_size_src,     # Dimensionality of sensor value u(x_i) (e.g., 1 for scalar)
                 input_size_tgt,      # Dimensionality of trunk input y (e.g., 1 for 1D)
                 output_size_tgt,     # Dimensionality of final output G(u)(y) (e.g., 1 for scalar)
                 p=32,                # Latent dimension for the branch/trunk cross product (Default)
                 phi_hidden_size=256, # Hidden layer size for the phi network (Default)
                 rho_hidden_size=256, # Hidden layer size for the rho network (Default)
                 trunk_hidden_size=256,# Hidden layer size for the trunk network (MLP) (Default)
                 n_trunk_layers=4,    # Number of layers in the trunk network (Default)
                 activation_fn=nn.ReLU, # Activation function
                 use_deeponet_bias=True, # Whether to use a bias term after the cross product
                 phi_output_size=128, # Output dimension of the phi network before aggregation (Default)
                 initial_lr=5e-4,     # Initial learning rate
                 lr_schedule_steps=None, # Steps for LR decay (optional)
                 lr_schedule_gammas=None, # Multiplicative factor for each step (optional)
                 use_positional_encoding=True, # Flag to enable/disable positional encoding
                 pos_encoding_dim=64, # Dimension for the positional encoding MLP output (concatenate) or sinusoidal input (film)
                 pos_encoding_type='skip', # Type: 'mlp', 'sinusoidal', or 'skip' (Default changed to skip)
                 pos_encoding_max_freq=1000.0, # Max frequency/scale for sinusoidal encoding
                 encoding_strategy='concatenate', # 'concatenate' or 'film'
                 film_modulation_dim=None, # Dimension for FiLM modulation (defaults to phi_hidden_size)
                 aggregation_type: str = "mean",  # 'mean' or 'attention'
                 attention_n_tokens: int = 8,     # k – number of learnable query tokens
                 ):
        super().__init__()

        # set hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt
        # Note: use_positional_encoding is True if type is 'mlp' or 'sinusoidal'
        self.use_positional_encoding = use_positional_encoding and pos_encoding_type != 'skip'
        self.pos_encoding_dim = pos_encoding_dim if self.use_positional_encoding else 0
        self.pos_encoding_type = pos_encoding_type
        self.pos_encoding_max_freq = pos_encoding_max_freq # Store max frequency/scale
        self.encoding_strategy = encoding_strategy
        # Default film_modulation_dim to phi_hidden_size if not provided
        self.film_modulation_dim = film_modulation_dim if film_modulation_dim is not None else phi_hidden_size

        self.p = p
        self.phi_hidden_size = phi_hidden_size
        self.phi_output_size = phi_output_size
        self.rho_hidden_size = rho_hidden_size
        self.trunk_hidden_size = trunk_hidden_size
        self.n_trunk_layers = n_trunk_layers

        # ---------------------------------------------------------------------
        # Aggregation choice ('mean' | 'attention')
        # ---------------------------------------------------------------------
        self.aggregation = aggregation_type.lower()
        if self.aggregation not in ["mean", "attention"]:
            raise ValueError("aggregation_type must be either 'mean' or 'attention'")
        self.attention_n_tokens = attention_n_tokens

        if self.aggregation == "attention":
            from utils.attention_pool import AttentionPool
            self.pool = AttentionPool(phi_output_size,
                                      n_heads=4,
                                      n_tokens=self.attention_n_tokens)

        # Removed function encoder parameter assignments
        # self.function_encoder_model = function_encoder_model
        # self.fe_n_basis = fe_n_basis
        # self.fe_concat_mode = fe_concat_mode
        # self.fe_arch = fe_arch

        # Validate encoding strategy
        if self.encoding_strategy not in ['concatenate', 'film']: # Removed 'function_encoder'
            raise ValueError("encoding_strategy must be 'concatenate' or 'film'")
        # Validate pos_encoding_type
        if self.pos_encoding_type not in ['mlp', 'sinusoidal', 'skip']:
             raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}. Choose 'mlp', 'sinusoidal', or 'skip'.")

        # Store LR schedule parameters
        self.initial_lr = initial_lr
        self.lr_schedule_steps = None
        self.lr_schedule_rates = None
        self.lr_schedule_gammas = None # Store gammas as well
        if lr_schedule_steps is not None:
            if lr_schedule_gammas is None or len(lr_schedule_steps) != len(lr_schedule_gammas):
                raise ValueError("lr_schedule_gammas must be provided and have the same length as lr_schedule_steps if scheduling is used.")
            self.lr_schedule_steps = sorted(lr_schedule_steps) # Ensure steps are sorted
            self.lr_schedule_gammas = lr_schedule_gammas # Store the provided gammas
            # Calculate the actual learning rates at each step based on gammas
            self.lr_schedule_rates = [initial_lr]
            current_lr = initial_lr
            for gamma in lr_schedule_gammas:
                current_lr *= gamma
                self.lr_schedule_rates.append(current_lr)

        # --- Branch Network (Deep Sets) ---
        self.pos_encoder_mlp = None # For concatenate strategy with MLP encoding
        self.film_pos_encoder = None # For film strategy (takes raw pos, sinusoidal feats, or skipped)
        self.u_projector = None # For film strategy

        if self.encoding_strategy == 'concatenate':
            # Determine phi input dimension and setup encoding components for concatenation
            if self.use_positional_encoding: # This is True only if type is 'mlp' or 'sinusoidal'
                phi_input_dim = self.pos_encoding_dim + output_size_src
                if self.pos_encoding_type == 'mlp':
                    # Simple MLP to encode positions - Added one more hidden layer
                    self.pos_encoder_mlp = nn.Sequential(
                        nn.Linear(input_size_src, self.pos_encoding_dim // 2),
                        activation_fn(),
                        # --- Added Layer ---
                        nn.Linear(self.pos_encoding_dim // 2, self.pos_encoding_dim // 2),
                        activation_fn(),
                        # --- End Added Layer ---
                        nn.Linear(self.pos_encoding_dim // 2, self.pos_encoding_dim)
                        # Consider adding LayerNorm or other activations if needed
                    )
                elif self.pos_encoding_type == 'sinusoidal':
                     # Ensure pos_encoding_dim is even and divisible by 2*input_size_src
                    if self.pos_encoding_dim % (2 * self.input_size_src) != 0:
                         raise ValueError(f"For sinusoidal encoding, pos_encoding_dim ({self.pos_encoding_dim}) must be divisible by 2 * input_size_src ({2 * self.input_size_src}).")
                    # No separate encoder MLP needed here for concat, sinusoidal feats used directly
                # No 'skip' case here because use_positional_encoding would be False
                else:
                    # This should not be reachable if use_positional_encoding logic is correct
                    raise ValueError(f"Invalid pos_encoding_type '{self.pos_encoding_type}' when use_positional_encoding is True.")
            else: # This case handles pos_encoding_type == 'skip' or if use_positional_encoding was explicitly False
                # Original input: raw position + sensor value
                phi_input_dim = input_size_src + output_size_src

            # Phi network: processes concatenated (encoded_location, value) or (location, value) pairs
            self.phi = nn.Sequential(
                nn.Linear(phi_input_dim, phi_hidden_size),
                activation_fn(),
                nn.Linear(phi_hidden_size, phi_hidden_size),
                activation_fn(),
                nn.Linear(phi_hidden_size, phi_output_size)
            )

        elif self.encoding_strategy == 'film':
            # Setup components for FiLM
            self.u_projector = nn.Linear(output_size_src, self.film_modulation_dim)

            film_encoder_input_dim = 0
            # Determine the input dimension for the FiLM parameter generator MLP
            if self.pos_encoding_type == 'mlp' or self.pos_encoding_type == 'skip':
                # FiLM encoder takes raw position and outputs gamma/beta
                film_encoder_input_dim = input_size_src
            elif self.pos_encoding_type == 'sinusoidal':
                # Ensure pos_encoding_dim is valid for sinusoidal
                if not self.use_positional_encoding:
                     raise ValueError("Sinusoidal pos_encoding_type requires use_positional_encoding=True (cannot be used with 'skip').")
                if self.pos_encoding_dim % (2 * self.input_size_src) != 0:
                     raise ValueError(f"For sinusoidal encoding, pos_encoding_dim ({self.pos_encoding_dim}) must be divisible by 2 * input_size_src ({2 * self.input_size_src}).")
                # FiLM encoder takes sinusoidal features and outputs gamma/beta
                film_encoder_input_dim = self.pos_encoding_dim
            else:
                 # Should be unreachable due to initial validation
                 raise ValueError(f"Unknown pos_encoding_type for FiLM: {self.pos_encoding_type}")

            # MLP to generate FiLM parameters (gamma, beta) from positional features
            # Output size is 2 * film_modulation_dim (for gamma and beta)
            self.film_pos_encoder = nn.Sequential(
                nn.Linear(film_encoder_input_dim, phi_hidden_size), # Input layer
                activation_fn(),
                nn.Linear(phi_hidden_size, phi_hidden_size), # Added hidden layer
                activation_fn(),                             # Added activation
                nn.Linear(phi_hidden_size, 2 * self.film_modulation_dim) # Output layer
            )

            # Phi network (core): processes the modulated sensor value 'u'
            # Input dim is film_modulation_dim (output of FiLM)
            # Output dim is phi_output_size
            self.phi = nn.Sequential(
                # The FiLM operation effectively acts as the first transformation
                nn.Linear(self.film_modulation_dim, phi_hidden_size), # First hidden layer
                activation_fn(),
                nn.Linear(phi_hidden_size, phi_hidden_size), # Second hidden layer
                activation_fn(),
                nn.Linear(phi_hidden_size, phi_output_size) # Output layer
            )

        # Rho network: processes aggregated representation from phi
        rho_input_dim = (self.phi_output_size *
                         (self.attention_n_tokens if self.aggregation == "attention" else 1))

        self.rho = nn.Sequential(
            nn.Linear(rho_input_dim, rho_hidden_size),
            activation_fn(),
            nn.Linear(rho_hidden_size, output_size_tgt * p)
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

        # create optimizer with the initial learning rate
        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr)

        # Initialize step counter for LR scheduling
        self.total_steps = 0

        # holdovers from function encoder code, these do nothing
        self.method = "deepOSet"
        self.average_function = None

        # Removed function encoder initialization logic
        # if self.encoding_strategy == 'function_encoder' and self.function_encoder_model is None:
            # ... (removed code) ...

    def _sinusoidal_encoding(self, coords):
        """Applies fixed sinusoidal encoding to coordinates."""
        # coords shape: (batch_size, n_sensors, input_size_src)
        # Output shape: (batch_size, n_sensors, pos_encoding_dim)

        # Ensure pos_encoding_dim is divisible by 2*input_size_src (already checked in init)
        dims_per_coord = self.pos_encoding_dim // self.input_size_src
        half_dim = dims_per_coord // 2

        # Frequency bands
        # Shape: (half_dim,)
        div_term = torch.exp(torch.arange(half_dim, device=coords.device) * -(torch.log(torch.tensor(self.pos_encoding_max_freq, device=coords.device)) / half_dim))

        # Expand div_term for broadcasting: (1, 1, 1, half_dim)
        div_term = div_term.view(1, 1, 1, half_dim)

        # Expand coords for broadcasting: (batch, n_sensors, input_size_src, 1)
        coords_expanded = coords.unsqueeze(-1)

        # Calculate arguments for sin/cos: (batch, n_sensors, input_size_src, half_dim)
        angles = coords_expanded * div_term

        # Calculate sin and cos embeddings: (batch, n_sensors, input_size_src, half_dim)
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)

        # Interleave sin and cos and flatten the last two dimensions
        # Shape: (batch, n_sensors, input_size_src, dims_per_coord)
        encoding = torch.cat([sin_embed, cos_embed], dim=-1).view(
            coords.shape[0], coords.shape[1], self.input_size_src, dims_per_coord
        )

        # Reshape to final desired dimension: (batch, n_sensors, pos_encoding_dim)
        encoding = encoding.view(coords.shape[0], coords.shape[1], self.pos_encoding_dim)

        return encoding

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

        # Reshape inputs for element-wise processing
        # Shape: (batch * n_sensors, *)
        xs_reshaped = xs.view(batch_size * n_sensors, self.input_size_src)
        us_reshaped = us.view(batch_size * n_sensors, self.output_size_src)

        phi_input_reshaped = None

        # --- Apply Encoding Strategy ---
        if self.encoding_strategy == 'concatenate':
            encoded_xs = None # Initialize variable to hold encoded xs
            if self.use_positional_encoding: # True only for 'mlp' or 'sinusoidal' types
                # --- Corrected Logic ---
                if self.pos_encoding_type == 'mlp':
                    if self.pos_encoder_mlp is None: # Check specific to MLP type
                        raise RuntimeError("Encoding strategy is 'concatenate' with 'mlp' but pos_encoder_mlp is not initialized.")
                    # Encode positions using MLP: (batch * n_sensors, pos_encoding_dim)
                    encoded_xs = self.pos_encoder_mlp(xs_reshaped)
                elif self.pos_encoding_type == 'sinusoidal':
                    # Calculate sinusoidal encoding: (batch, n_sensors, pos_encoding_dim)
                    encoded_xs_full = self._sinusoidal_encoding(xs)
                    # Reshape: (batch * n_sensors, pos_encoding_dim)
                    encoded_xs = encoded_xs_full.view(batch_size * n_sensors, self.pos_encoding_dim)
                else:
                    # Should not happen if __init__ validation is correct
                    raise ValueError(f"Invalid pos_encoding_type '{self.pos_encoding_type}' encountered in forward_branch when use_positional_encoding is True.")

                # Concatenate the correctly encoded location and value
                # Shape: (batch * n_sensors, pos_encoding_dim + output_size_src)
                phi_input_reshaped = torch.cat((encoded_xs, us_reshaped), dim=1)
                # --- End Corrected Logic ---
            else: # Handles 'skip' type or explicitly disabled positional encoding
                # Original: Concatenate raw location and value
                # Shape: (batch * n_sensors, input_size_src + output_size_src)
                phi_input_reshaped = torch.cat((xs_reshaped, us_reshaped), dim=1)

        elif self.encoding_strategy == 'film':
            if self.film_pos_encoder is None or self.u_projector is None:
                 raise RuntimeError("Encoding strategy is 'film' but FiLM components are not initialized.")

            pos_features_reshaped = None
            # Determine input features for the FiLM parameter generator
            if self.pos_encoding_type == 'mlp' or self.pos_encoding_type == 'skip':
                # Use raw positions as input to the FiLM parameter generator
                pos_features_reshaped = xs_reshaped
            elif self.pos_encoding_type == 'sinusoidal':
                 # Check if encoding is actually enabled (it should be for sinusoidal)
                if not self.use_positional_encoding:
                     raise RuntimeError("Sinusoidal encoding type selected for FiLM, but use_positional_encoding is False.")
                # Calculate sinusoidal encoding first
                # Shape: (batch, n_sensors, pos_encoding_dim)
                encoded_xs_full = self._sinusoidal_encoding(xs)
                # Reshape: (batch * n_sensors, pos_encoding_dim)
                pos_features_reshaped = encoded_xs_full.view(batch_size * n_sensors, self.pos_encoding_dim)
            # No else needed due to __init__ validation

            # Generate FiLM parameters (gamma, beta)
            # Shape: (batch * n_sensors, 2 * film_modulation_dim)
            gamma_beta = self.film_pos_encoder(pos_features_reshaped)
            # Split into gamma and beta
            # Shape: (batch * n_sensors, film_modulation_dim) each
            gamma, beta = torch.split(gamma_beta, self.film_modulation_dim, dim=1)

            # Project sensor values 'u'
            # Shape: (batch * n_sensors, film_modulation_dim)
            u_proj = self.u_projector(us_reshaped)

            # Apply FiLM: modulated_u = gamma * u_proj + beta
            # Shape: (batch * n_sensors, film_modulation_dim)
            phi_input_reshaped = gamma * u_proj + beta

        else:
            # This should now be unreachable due to __init__ validation
            raise ValueError(f"Unknown encoding_strategy: {self.encoding_strategy}")
        # --- End Encoding Strategy ---

        # Apply phi network (or phi_core for FiLM)
        # Input shape: (batch_size * n_sensors, phi_input_dim or film_modulation_dim)
        # Output shape: (batch_size * n_sensors, phi_output_size)
        phi_output = self.phi(phi_input_reshaped)

        # Reshape back for aggregation
        # Shape: (batch_size, n_sensors, phi_output_size)
        phi_output_reshaped = phi_output.view(batch_size, n_sensors, self.phi_output_size)

        # ---- Aggregation over sensors ---------------------------------------
        if self.aggregation == "mean":
            aggregated = torch.mean(phi_output_reshaped, dim=1)            # (B, dφ)
        else:  # attention
            aggregated = self.pool(phi_output_reshaped)                    # (B, dφ)

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

    def _get_current_lr(self):
        """ Gets the learning rate based on the current total_steps. """
        # If no schedule is defined, always return the initial LR
        if self.lr_schedule_steps is None or self.lr_schedule_rates is None:
            return self.initial_lr

        lr = self.initial_lr
        # Find the correct LR based on the number of steps completed
        milestone_idx = -1
        for i, step_milestone in enumerate(self.lr_schedule_steps):
            if self.total_steps >= step_milestone:
                milestone_idx = i
            else:
                break # Stop checking once we are below a milestone

        # If we passed any milestones, use the corresponding rate
        if milestone_idx != -1:
             # +1 because lr_schedule_rates[0] is initial LR
            lr = self.lr_schedule_rates[milestone_idx + 1]
        return lr

    def _update_lr(self):
        """ Updates the optimizer's learning rate based on total_steps. """
        new_lr = self._get_current_lr()
        # Update learning rate for all parameter groups in the optimizer
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr
        return new_lr # Return the new LR for logging/display

    # This is the main training loop, kept consistent with the function encoder code.
    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int, # Note: This loop runs 'epochs' times, each is one step
                    progress_bar=True,
                    callback: BaseCallback = None):
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            # Pass initial state including the step counter if needed
            callback.on_training_start(locals())


        losses = []
        # Treat 'epochs' here as the number of steps for this training call
        bar = trange(epochs) if progress_bar else range(epochs)
        for step_in_epoch in bar: # Renamed 'epoch' to 'step_in_epoch' for clarity
            # Update Learning Rate based on total steps *before* optimizer step
            current_lr = self._update_lr()

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

            # Increment total steps *after* optimizer step
            self.total_steps += 1

            # update progress bar
            if progress_bar:
                # Display current LR in the progress bar
                bar.set_description(f"Step {self.total_steps} | Loss: {loss.item():.4e} | Grad Norm: {norm:.2f} | LR: {current_lr:.2e}")

            # callbacks
            if callback is not None:
                # Pass current state including step counter and LR
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
        params["phi_output_size"] = self.phi_output_size
        params["rho_hidden_size"] = self.rho_hidden_size
        params["trunk_hidden_size"] = self.trunk_hidden_size
        params["n_trunk_layers"] = self.n_trunk_layers
        # params["activation_fn"] = self.phi[1].__class__.__name__ # Get activation class name
        params["use_deeponet_bias"] = self.bias is not None
        # Add LR schedule params if defined
        params["initial_lr"] = self.initial_lr
        if self.lr_schedule_steps is not None:
            params["lr_schedule_steps"] = str(self.lr_schedule_steps)
        if self.lr_schedule_gammas is not None:
             params["lr_schedule_gammas"] = str(self.lr_schedule_gammas)
        # Add positional encoding params
        params["use_positional_encoding"] = self.use_positional_encoding # Reflects if MLP/Sinusoidal is active
        # Always log the type, even if 'skip'
        params["pos_encoding_type"] = self.pos_encoding_type
        if self.use_positional_encoding: # Only log dim/freq if encoding is active
            params["pos_encoding_dim"] = self.pos_encoding_dim
            if self.pos_encoding_type == 'sinusoidal':
                params["pos_encoding_max_freq"] = self.pos_encoding_max_freq
        # Add encoding strategy params
        params["encoding_strategy"] = self.encoding_strategy
        if self.encoding_strategy == 'film':
            params["film_modulation_dim"] = self.film_modulation_dim

        # Aggregation logging
        params["aggregation_type"] = self.aggregation
        if self.aggregation == "attention":
            params["attention_n_tokens"] = self.attention_n_tokens

        # Removed function encoder params from logging
        # if self.encoding_strategy == 'function_encoder':
        #     params["fe_n_basis"] = self.fe_n_basis
        #     params["fe_concat_mode"] = self.fe_concat_mode
        #     params["fe_arch"] = self.fe_arch

        params = {k: str(v) for k, v in params.items()}
        return params
# tot totoot to curururu