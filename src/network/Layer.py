import numpy as np

from .activation import ReLu

class SidLinearLayer:
    def __init__(
        self,
        input: int,
        output: int,
        random_weights: bool = True
    ):
        self.shape = (input, output)
        if random_weights:
            self.weights = np.random.randn(*self.shape)
            self.bias = np.random.randn(output)
        else:
            self.weights = np.zeros(*self.shape)
            self.bias = np.zeros(output)
        
        self.last_forward = None
        self.last_input = None

    def forward(
        self,
        x: np.ndarray
    ):
        self.last_input = x
        self.last_forward = x @ self.weights
        return self.last_forward

    def backward(
        self,
        gradient: np.ndarray,
        lr: float = 0.01
    ):
        """L = f'f(x) + f'(x)
        """

        update = self.last_input.T @ gradient # (gradient)
        self.weights -= update * lr

        return gradient @ self.weights.T # self.weights @ gradient