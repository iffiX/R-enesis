import torch


class Interpolator:
    def __init__(self, steps, start_x, stop_x, start_y, stop_y):
        self.steps = steps
        self.start_x = start_x
        self.stop_x = stop_x
        self.start_y = start_y
        self.stop_y = stop_y

    def __call__(self, x):
        if self.start_y == self.stop_y:
            return torch.zeros_like(x)
        else:
            return torch.log(
                (x / self.steps - self.start_x)
                / (self.stop_x - self.start_x)
                * (self.stop_y - self.start_y)
                + self.start_y
                + 1e-10
            )
