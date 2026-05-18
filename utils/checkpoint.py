import numpy as np

def save_global(path, w_vec: np.ndarray, t: int, rng_state):
    np.savez(path, w=w_vec, t=t, rng=rng_state)

def load_global(path):
    z = np.load(path, allow_pickle=True)
    return z["w"], int(z["t"]), z["rng"].item()
