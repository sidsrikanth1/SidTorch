from src.builder import SidNetwork
from src.network.Layer import SidLinearLayer
from src.network.activation import ReLu
from src.training.dataset import addition_dataset

import matplotlib.pyplot as plt
import numpy as np
import tqdm

if __name__ == "__main__":
    N = 1000
    d = 8

    # create MLP (default activation = relu)
    MLP = SidNetwork(
        layers = [
            SidLinearLayer(
                input=d,
                output=d,
            ),
            SidLinearLayer(
                input=d,
                output=1,
            ),
        ]
    )

    
    X_train, y_train = addition_dataset(
        d,
        N,
        low=-5,
        high=5
    )

    # iterate through each training example
    loss_graph = []
    for epoch in tqdm.trange(1000):
        y_pred = MLP.forward(X_train)
        y_pred = y_pred.reshape(-1, 1)

        # backpropagation
        MLP.backward(
            y_pred,
            y_train,
            lr=0.0001
        )
        
        loss_graph.append(
            np.sum((y_pred - y_train) ** 2)
        )

    plt.plot(loss_graph)