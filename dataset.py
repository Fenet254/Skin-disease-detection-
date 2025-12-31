import numpy as np

def load_dataset():
    # Dummy dataset (replace with real medical data later)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)  # 0 = healthy, 1 = disease
    return X, y
