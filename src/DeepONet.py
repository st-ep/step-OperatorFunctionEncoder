import torch
import torch.nn.functional as F   #  <-- NEW
from FunctionEncoder import BaseDataset, BaseCallback
from tqdm import trange
from torch.optim.lr_scheduler import _LRScheduler # Import base class for type hinting if needed


# This implements an unstacked DeepONet
class DeepONet(torch.nn.Module):

    def __init__(self,
                 input_size_src, # the dimensionality of the inputs to u. In the paper it is always 1, but we can be more general.
                 output_size_src, # the dimensionality of the output of u. In the paper it is always 1, but we can be more general.
                 input_size_tgt, # the dimensionality y. In the paper it is always 1, but we can be more general.
                 output_size_tgt, # the dimensionality of the output of y. In the paper it is always 1, but we can be more general.
                 n_input_sensors, # the number of input sensors, "m" in the paper
                 p=20, # This is the number of terms for the final dot product operation. In the paper, they say at least 10.
                 use_deeponet_bias=True, # whether to use a bias term after the cross product in the deepnet.
                 hidden_size=256,
                 n_layers=4,
                 initial_lr=5e-4,     # Initial learning rate (matches DeepOSet default)
                 lr_schedule_steps=None, # Steps for LR decay (optional)
                 lr_schedule_gammas=None, # Multiplicative factor for each step (optional)
                 weight_decay=0.0,           # Weight decay for Adam
                 interpolate_sensor_gaps=True  # <‑‑ NEW FLAG
                 ):
        super().__init__()

        # set hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt

        self.n_input_sensors = n_input_sensors
        self.p = p
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Store LR schedule parameters
        self.weight_decay = weight_decay
        self.initial_lr = initial_lr
        self.lr_schedule_steps = None
        self.lr_schedule_rates = None
        self.lr_schedule_gammas = None # Store gammas as well
        self.interpolate_sensor_gaps = interpolate_sensor_gaps   # <‑‑ STORE
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

        # this maps u(x_1), u(x_2), ..., u(x_m) to b_1, b_2, ..., b_p
        layers_branch = []
        layers_branch.append(torch.nn.Linear(output_size_src * n_input_sensors, hidden_size))
        layers_branch.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            layers_branch.append(torch.nn.Linear(hidden_size, hidden_size))
            layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.Linear(hidden_size, output_size_tgt * p))
        self.branch = torch.nn.Sequential(*layers_branch)

        # this maps y to t_1, ..., t_p
        trunk_layers = []
        trunk_layers.append(torch.nn.Linear(input_size_tgt, hidden_size))
        trunk_layers.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            trunk_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(torch.nn.ReLU())
        trunk_layers.append(torch.nn.Linear(hidden_size, output_size_tgt * p))
        trunk_layers.append(torch.nn.Sigmoid())
        self.trunk = torch.nn.Sequential(*trunk_layers)

        # an optional bias, see equation 2 in the paper.
        self.bias = torch.nn.Parameter(torch.randn(output_size_tgt) * 0.1) if use_deeponet_bias else None

        # create optimizer with the initial learning rate
        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)

        # Initialize step counter for LR scheduling
        self.total_steps = 0

        # holdovers from function encoder code, these do nothing
        self.method = "deepONet"
        self.average_function = None

        self._last_interpolated_u    = None    # for plotting
        self._last_interpolated_mask = None
        self._sensor_positions       = None    # saved once (training grid)

    def forward_branch(self, u, xs_pos):
        """
        Compute branch output B(u) while gracefully handling cases where the
        number of provided sensors differs from the number used in training
        (self.n_input_sensors).

        If fewer sensors are given, we pad the missing ones by a simple
        interpolation strategy: fill with the mean value of the available
        sensors.  If more sensors are supplied we truncate the extras.
        """
        batch_size, n_sensors_current, _ = u.shape

        # ---------------------------------------------------------------
        # EARLY EXIT if user disabled interpolation / resampling
        # ---------------------------------------------------------------
        if not self.interpolate_sensor_gaps:
            if n_sensors_current != self.n_input_sensors:
                raise ValueError(
                    f"DeepONet: got {n_sensors_current} sensors but "
                    f"interpolate_sensor_gaps=False (expected "
                    f"{self.n_input_sensors}).  Either supply exactly the "
                    f"training count or enable interpolation."
                )

            # -----------------------------------------------------------
            # Save reference grid *once* so that plotting functions can
            # access it later (needed for red‑X markers).
            # -----------------------------------------------------------
            if (
                self._sensor_positions is None
                and xs_pos is not None
                and xs_pos.shape[1] == self.n_input_sensors
            ):
                # assume xs_pos identical across batch
                self._sensor_positions = xs_pos[0].detach().clone()

            # record empty mask for plotting
            missing_mask = torch.zeros(
                batch_size, self.n_input_sensors, dtype=torch.bool,
                device=u.device
            )
            self._last_interpolated_u    = u.detach()
            self._last_interpolated_mask = missing_mask

            ins  = u.reshape(batch_size, -1)
            outs = self.branch(ins).reshape(batch_size, -1, self.output_size_tgt)
            return outs

        # ---------------------------------------------------------------
        # Save the _reference_ grid **only** if we see the complete grid
        # ---------------------------------------------------------------
        if (
            self._sensor_positions is None
            and xs_pos is not None
            and xs_pos.shape[1] == self.n_input_sensors
        ):
            # assume xs_pos is identical for every sample in the batch
            self._sensor_positions = xs_pos[0].detach().clone()  # (m, input_dim)

        # ------------------------------------------------------------------
        # 1‑D linear interpolation / resampling along the sensor dimension.
        # If the number of sensors differs from training, we resample u so
        # that it has exactly `self.n_input_sensors` entries.  This means
        # every "hole" is filled with the mean of its neighbouring sensors.
        # ------------------------------------------------------------------
        if n_sensors_current == self.n_input_sensors:
            # nothing to do
            missing_mask = torch.zeros(
                batch_size, self.n_input_sensors, dtype=torch.bool,
                device=u.device
            )

        elif (
            self._sensor_positions is None
            or self._sensor_positions.shape[0] != self.n_input_sensors
        ):
            # ----------------------------------------------------------------
            # Fallback: we don't know / don't trust full grid, use 1‑D
            # interpolation exactly like the earlier simple version.
            # ----------------------------------------------------------------
            u = u.permute(0, 2, 1)  # (B, C, L)
            mode = "linear" if n_sensors_current > 1 else "nearest"
            u = F.interpolate(
                u,
                size=self.n_input_sensors,
                mode=mode,
                align_corners=False if mode == "linear" else None,
            ).permute(0, 2, 1)       # (B, m, C)

            # we mark *all* padded positions as "missing"
            # (use batch dimension so that shape is always (B , m))
            missing_mask = torch.ones(
                batch_size, self.n_input_sensors, dtype=torch.bool,
                device=u.device
            )

        else:
            # ---------- robust index‑based mapping -------------------------
            m = self.n_input_sensors
            full_u       = u.new_empty(batch_size, m, self.output_size_src)
            missing_mask = torch.zeros(batch_size, m, dtype=torch.bool,
                                       device=u.device)

            # reference grid & helpful sorted order
            ref_x  = self._sensor_positions.squeeze(-1)        # (m,)
            order  = torch.argsort(ref_x)                      # ascending by x
            invord = torch.argsort(order)                      # to unsort later
            ref_x_sorted = ref_x[order]                       # (m,)

            for b in range(batch_size):
                # ---------- place available sensors ----------
                diff = torch.cdist(
                    self._sensor_positions.unsqueeze(0),
                    xs_pos[b].detach().unsqueeze(0)
                ).squeeze(0)                                   # (m , n_curr)
                nearest_ref = diff.argmin(dim=0)               # (n_curr,)

                full_u[b].fill_(float('nan'))
                full_u[b].index_copy_(0, nearest_ref, u[b])    # put known sensors

                # ---------- interpolate missing (sorted grid) ----------
                fu_sorted = full_u[b][order]                   # (m, C) sorted
                present   = ~torch.isnan(fu_sorted[:, 0])
                missing   = ~present

                # remember mask in *original* order
                missing_mask[b] = missing.clone()[invord]

                if missing.any():
                    pres_idx = torch.where(present)[0]
                    for mi in torch.where(missing)[0]:
                        # left/right neighbour indices among PRESENT points
                        left_idx  = pres_idx[pres_idx < mi].max() if (pres_idx < mi).any() else pres_idx.min()
                        right_idx = pres_idx[pres_idx > mi].min() if (pres_idx > mi).any() else pres_idx.max()
                        xl, xr = ref_x_sorted[left_idx], ref_x_sorted[right_idx]
                        t = (ref_x_sorted[mi] - xl) / (xr - xl + 1e-12)
                        fu_sorted[mi] = (1 - t) * fu_sorted[left_idx] + t * fu_sorted[right_idx]

                # unsort to original grid
                full_u[b] = fu_sorted[invord]

            u = full_u

        # -------------  Store for later visualisation ------------------
        self._last_interpolated_u    = u.detach()
        self._last_interpolated_mask = missing_mask.detach()   # (B , m)

        # Flat vector for the branch MLP
        ins = u.reshape(batch_size, -1)       # (B, n_input_sensors*output_size_src)
        outs = self.branch(ins)
        outs = outs.reshape(batch_size, -1, self.output_size_tgt)
        return outs

    def forward_trunk(self, y):
        outs = self.trunk(y)
        outs = outs.reshape(outs.shape[0], y.shape[1], -1, self.output_size_tgt)
        return outs

    def forward(self, xs, us, ys):
        # xs are not actually used for deeponet, but we keep them to be consistent with the function encoder
        # us are the values of u at the input sensors
        # ys are the locations of the output sensors.
        b = self.forward_branch(us, xs)
        t = self.forward_trunk(ys)

        # this is just the dot product, but allowing for the output dim to be > 1
        G_u_y = torch.einsum("fpz,fdpz->fdz", b, t)

        # optionally add bias
        if self.bias is not None:
            G_u_y = G_u_y + self.bias

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

            # sample input data
            xs, u_xs, ys, G_u_ys, _ = dataset.sample(device=device)


            # approximate functions, compute error
            estimated_G_u_ys = self.forward(xs, u_xs, ys)
            prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)

            # add loss components
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
        params["input_size_src"] = self.input_size_src
        params["output_size_src"] = self.output_size_src
        params["input_size_tgt"] = self.input_size_tgt
        params["output_size_tgt"] = self.output_size_tgt
        params["n_input_sensors"] = self.n_input_sensors
        params["p"] = self.p
        params["hidden_size"] = self.hidden_size
        params["use_deeponet_bias"] = self.bias is not None
        params["initial_lr"] = self.initial_lr
        if self.lr_schedule_steps is not None:
            params["lr_schedule_steps"] = str(self.lr_schedule_steps)
        if self.lr_schedule_gammas is not None:
             params["lr_schedule_gammas"] = str(self.lr_schedule_gammas)
        params["weight_decay"] = self.weight_decay
        params["interpolate_sensor_gaps"] = self.interpolate_sensor_gaps
        params = {k: str(v) for k, v in params.items()}
        return params

    # ------------------------------------------------------------------
    # Information for plotting (called after a forward pass)
    # ------------------------------------------------------------------
    def get_last_interpolation_info(self):
        if self._last_interpolated_u is None:
            return None
        return {
            "full_u":   self._last_interpolated_u,       # (B, m, out_dim)
            "mask":     self._last_interpolated_mask,    # (B , m)
            "sensor_xs":self._sensor_positions           # (m, input_dim)
        }

