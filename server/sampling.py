import numpy as np
def sample_clients(K: int, m: int, rng: np.random.Generator):
    m = min(m, K)
    return rng.choice(K, size=m, replace=False).tolist()
