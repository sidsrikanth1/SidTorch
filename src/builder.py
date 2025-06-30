import numpy as np

class SidNetwork:
    def __init__(
        self,
        layers: list
    ):
        self.layers = layers
            
    def forward(
        self,
        x: np.ndarray
    ):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(
        self,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        lr: float = 0.01
    ):
        """Forward looks something like f(f(f(..)))
        
        The final layer should apply the chain rule, i.e. f'(f(f(..))) * d/dx(f(f(..))
        
        To do this, we can propogate the gradient and continue to multiply it with the
        previous gradients as we reach the final layer.
        
        first_layer: Loss
        second_layer: df/dx(first_layer) * first_layer
        curr_layer: df/dx(prev_layer) * prev_layer
        """
        # ensure predictions are in the same shape as training
        N = len(y_train) 

        y_pred = y_pred.reshape(y_train.shape)
        gradient = 2 * (y_pred - y_train) / N

        for layer in self.layers[::-1]:
            backward = layer.backward(
                gradient,
                lr=lr
            )
            gradient = backward  # get the net gradient from going backwards