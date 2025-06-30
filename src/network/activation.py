import numpy as np


class Activation:
    """Abstract base class for activation
    """
    def __init__(self):
        pass


class ReLu(Activation):
    def forward(
        self,
        x: np.ndarray
    ):
        # return np.matmul(x, np.diag(x.flatten() > 0))
        return np.maximum(x,0)

    def gradient(
        self,
        x: np.ndarray
    ):
        return (x > 0).astype(np.float32)