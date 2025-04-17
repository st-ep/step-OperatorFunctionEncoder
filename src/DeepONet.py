import torch
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
                 lr_schedule_gammas=None # Multiplicative factor for each step (optional)
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
        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr)

        # Initialize step counter for LR scheduling
        self.total_steps = 0

        # holdovers from function encoder code, these do nothing
        self.method = "deepONet"
        self.average_function = None

    def forward_branch(self, u):
        ins = u.reshape(u.shape[0], -1)
        outs = self.branch(ins)
        outs = outs.reshape(outs.shape[0], -1, self.output_size_tgt)
        return outs

    def forward_trunk(self, y):
        outs = self.trunk(y)
        outs = outs.reshape(outs.shape[0], y.shape[1], -1, self.output_size_tgt)
        return outs

    def forward(self, xs, us, ys):
        # xs are not actually used for deeponet, but we keep them to be consistent with the function encoder
        # us are the values of u at the input sensors
        # ys are the locations of the output sensors.
        b = self.forward_branch(us)
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
        params = {k: str(v) for k, v in params.items()}
        return params

