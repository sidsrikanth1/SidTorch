import numpy as np

def addition_dataset(
    input_size: int,
    points: int,
    low: int = -50,
    high: int = 50
):
    """Random dataset generator (should be completely learnable)
    
    Returns: X_train, Y_train
    """
    
    # random nd integer array
    x = np.random.randint(low, high, size=(points, input_size))
    y = np.sum(x, axis=1).reshape(-1, 1)
    return x, y