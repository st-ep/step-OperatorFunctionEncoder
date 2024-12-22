from typing import Tuple, Union
from plotting_specs import colors, labels, titles

import torch
from matplotlib import pyplot as plt
from src.Datasets.OperatorDataset import OperatorDataset
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
from plotting_specs import colors, labels, titles

class QuadraticIntegralDataset(OperatorDataset):

    def __init__(self,
                 a_range=(-3/5, 3/5),
                 b_range=(-3/5, 3/5),
                 c_range=(-3/5, 3/5),
                 input_ranges=[(-10, 2), (8, 10)],
                 device="cuda",
                 *args,
                 **kwargs
                 ):
        super().__init__(input_size=(1,), output_size=(1,), *args, **kwargs)
        self.a_range = torch.tensor(a_range, dtype=torch.float32, device=device)
        self.b_range = torch.tensor(b_range, dtype=torch.float32, device=device)
        self.c_range = torch.tensor(c_range, dtype=torch.float32, device=device)
        self.input_ranges = [torch.tensor(r, dtype=torch.float32, device=device) for r in input_ranges]
        self.device = device


    # the info dict is used to generate data. So first we generate an info dict
    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        As = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32, device=self.device) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        Bs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32, device=self.device) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        Cs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32, device=self.device) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]

        return {"As": As, "Bs": Bs, "Cs": Cs}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        # Determine the number of samples per range
        n_ranges = len(self.input_ranges)
        samples_per_range = n_samples // n_ranges
        remainder = n_samples % n_ranges

        samples = []
        for i, r in enumerate(self.input_ranges):
            # Allocate remaining samples to the last range
            current_samples = samples_per_range + (1 if i < remainder else 0)
            if current_samples > 0:
                xs = torch.rand((info["As"].shape[0], current_samples, *self.input_size), dtype=torch.float32, device=self.device)
                xs = xs * (r[1] - r[0]) + r[0]
                samples.append(xs)

        # Concatenate samples from all ranges
        xs = torch.cat(samples, dim=1)
        
        # Shuffle the samples to ensure randomness
        indices = torch.randperm(xs.shape[1])
        xs = xs[:, indices]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        # returns the integral of a quadratic function ax^2 + bx + c
        As, Bs, Cs = info["As"], info["Bs"], info["Cs"]
        ys = 1/3. * As.unsqueeze(1) * inputs ** 3 + 1/2. * Bs.unsqueeze(1) * inputs ** 2 + Cs.unsqueeze(1) * inputs
        return ys