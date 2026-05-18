import numpy as np
def coord_median(U_list):
    return np.median(np.stack(U_list,0), axis=0)
def trimmed_mean(U_list, trim=0.1):
    U = np.stack(U_list,0); K = U.shape[0]
    lo = int(K*trim); hi = K-lo
    return np.sort(U, axis=0)[lo:hi].mean(axis=0)
