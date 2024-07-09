from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class IntegralDataset(BaseDataset):

    def __init__(self,
                 a_range=(-3, 3),
                 b_range=(-3, 3),
                 c_range=(-3, 3),
                 input_range=(-10, 10),
                 n_functions_per_sample:int = 10,
                 n_examples_per_sample:int = 1_000,
                 n_points_per_sample:int = 10_000,
                 freeze_xs:bool = False
                 ):
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="deterministic",
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         )
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.input_range = input_range
        self.freeze_xs = freeze_xs
        self.xs = None

    def sample(self, device:Union[str, torch.device] ="auto") -> Tuple[ torch.tensor,
                                                                torch.tensor,
                                                                torch.tensor,
                                                                torch.tensor,
                                                                dict]:
        with torch.no_grad():
            device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # generate n_functions sets of coefficients
            As = torch.rand((n_functions, 1), dtype=torch.float32) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
            Bs = torch.rand((n_functions, 1), dtype=torch.float32) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
            Cs = torch.rand((n_functions, 1), dtype=torch.float32) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]

            # generate n_samples_per_function samples for each function
            # if freezing inputs, this generates them at the first call, and otherwise loads them
            # otherwise, it generates them at each call
            if not self.freeze_xs or self.xs is None:
                example_xs = torch.rand((n_functions, n_examples, *self.input_size), dtype=torch.float32)
                example_xs = example_xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            else:
                example_xs = self.xs
            if self.freeze_xs and self.xs is None:
                self.xs = example_xs

            # these are always random
            xs = torch.rand((n_functions, n_points, *self.input_size), dtype=torch.float32)
            xs = xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]


            # compute the corresponding ys
            # example ys are the true function
            example_ys = As.unsqueeze(1) * example_xs ** 2 + Bs.unsqueeze(1) * example_xs + Cs.unsqueeze(1)

            # ys are the transformed function, in this the derivative
            transformed_ys = 1/3. * As.unsqueeze(1) * xs ** 3 + 1/2. * Bs.unsqueeze(1) * xs ** 2 + Cs.unsqueeze(1) * xs

            # move to device
            xs = xs.to(device)
            transformed_ys = transformed_ys.to(device)
            example_xs = example_xs.to(device)
            example_ys = example_ys.to(device)


            return example_xs, example_ys, xs, transformed_ys, {"As":As, "Bs" : Bs, "Cs": Cs}