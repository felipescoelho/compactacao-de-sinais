"""utils.py

Some utilitary methods for preprocessing and stuff.

luizfelipe.coelho@smt.ufrj.br
Jun 18, 2024
"""


import numpy as np


def dft_channel(H: np.ndarray, N: int):
    """Method to perform the channel DFT.
    
    Parameters
    ----------
    H : np.ndarray
        Channel matrix (N*L, num_chann, K)
    N : int
        Number of antennnas
    """

    NL, num_chann, K = H.shape
    L = int(NL/N)
    dft_mat = np.array([[np.exp(-1j*2*np.pi*n*k/N) for n in range(N)]
                        for k in range(N)], dtype=np.complex128)
    H_dft = np.zeros((NL, num_chann, K), dtype=np.complex128)

    for it in range(num_chann):
        for l in range(L):
            H_dft[l*N:(l+1)*N, it, :] = dft_mat @ H[l*N:(l+1)*N, it, :]
    
    return H_dft


def idft_channel(H: np.ndarray, N: int):
    """Method to perform the channel IDFT.
    
    Parameters
    ----------
    H : np.ndarray
        Channel matrix (N*L, num_chann, K)
    N : int
        Number of antennas
    """

    NL, num_chann, K = H.shape
    L = int(NL/N)
    idft_mat = np.array([[np.exp(1j*2*np.pi*n*k/N)/N for n in range(N)]
                        for k in range(N)], dtype=np.complex128)
    H_idft = np.zeros((NL, num_chann, K), dtype=np.complex128)

    for it in range(num_chann):
        for l in range(L):
            H_idft[l*N:(l+1)*N, it, :] = idft_mat @ H[l*N:(l+1)*N, it, :]
    
    return H_idft



def prepare_dataset(H: np.ndarray, N):
    """Method to prepare dataset for compression."""
    NL, num_chann, K = H.shape
    L = int(NL/N)
    X = np.zeros((N*K, num_chann, L), dtype=np.complex128)
    for it in range(num_chann):
        for l in range(L):
            X[:, it, l] = H[l*N:(l+1)*N, it, :].flatten()
    return X


def calculate_gzf(H: np.ndarray):
    """Method to calculate the Global Zero-Forcing precoding
    
    Parameters
    ----------
    H : np.ndarray
    """

    NL, num_chann, K = H.shape
    W = np.zeros((NL, K, num_chann), dtype=np.complex128)
    for it in range(num_chann):
        F = H[:, it, :] @ np.linalg.pinv(np.conj(H[:, it, :]).T @ H[:, it, :])
        for k in range(K):
            W[:, k, it] = F[:, k]/(np.linalg.norm(F[:, k])**2)
    
    return W


def calculate_bps(H: np.ndarray, W: np.ndarray, A: np.ndarray, sigma_n2: float):
    """Method to calculate the achievable bit rate
    
    Parameters
    ----------
    H : np.ndarray
        Channel matrix (NL, num_chann, K)
    W : np.ndarray
        Precoding matrix (NL, K, num_chann)
    A : np.ndarray
        Adjacency matrix (L, K)
    sigma_n2 : float
        Noise power
    """

    L, K = A.shape
    NL, num_chann, _ = H.shape
    N = int(NL/L)
    sinr = np.zeros((K, num_chann))
    r = np.zeros((K, num_chann))
    eta = 1/np.sum(A, axis=1)  # How much power is allocated to each user
    for it in range(num_chann):
        for k in range(K):
            Lk = np.nonzero(A[:, k])[0]
            spam = 0
            for l in Lk:
                h_lk = H[l*N:(l+1)*N, it, k]
                w_lk = W[l*N:(l+1)*N, k, it]
                spam += np.vdot(h_lk, w_lk)*np.sqrt(eta[l])
            saussage = 0
            for t in range(K):
                if t == k:
                    continue
                Ll = np.nonzero(A[:, t])[0]
                eggs = 0
                for l in Ll:
                    h_lk = H[l*N:(l+1)*N, it, k]
                    w_lt = W[l*N:(l+1)*N, t, it]
                    eggs += np.vdot(h_lk, w_lt)*np.sqrt(eta[l])
                saussage += np.abs(eggs)**2
            sinr[k, it] = np.abs(spam)**2 / (saussage + sigma_n2)
        r[:, it] = np.log2(1 + sinr[:, it])

    return sinr, r
