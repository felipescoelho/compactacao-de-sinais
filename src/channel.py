"""channel.py

Script to simulate the channel matrix of cell-free massive MIMO
communication system.

luizfelipe.coelho@smt.ufrj.br
Mar 20, 2024
"""


import numpy as np
from scipy.integrate import quad, dblquad
from scipy.linalg import toeplitz, sqrtm


def calc_K_factor(d_lk: float) -> float:
    """Method to calculate the K-factor.

    The K-factor indecates the dominance of the LoS component over the
    NLoS component [1].
    
    [1] L. Sun, J. Hou and T. Shu, "Bandwidth-Efficient Precoding in
        Cell-Free Massive MIMO Networks with Rician Fading Channels,"
        2021 18th Annual IEEE International Conference on Sensing,
        Communication, and Networking (SECON), Rome, Italy, 2021,
        pp. 1-9, doi: 10.1109/SECON52354.2021.9491585.
    
    Parameters
    ----------
    d_lk : float
        Distance between the l-th AP and the k-th UE, in meters.
    
    Returns
    -------
    K_lk : float
        K-factor for the channel between the l-th AP and the k-th UE, in
        dB.
    """

    K_lk = 10**(1.3 - .003*d_lk)

    return K_lk


def corr_mat_local_scatter(N: int, angle_varphi: float, angle_theta: float,
                           sigma_varphi: float, sigma_theta: float,
                           antenna_spacing: float) -> np.ndarray:
    """
    Method to generate the spacial correlation matrix for a given local
    scattering model.

    Parameters
    ----------
    N : int
        Number of antennas in each Access Point (AP)
    angle_varphi : float
        Nominal azimuth Angle of Arrival (AoA)
    angle_theta : float
        Nominal elevation AoA
    sigma_varphi : float
        Angular Standard Deviation (ASD) around nominal azimuth angle
    sigma_theta : float
        ASD around nominal elevation angle
    antenna_spacing : float
        Space between antannas of uniform linear array (ULA) in
        wavelengths

    Returns
    -------
    R : np.ndarray
        Spacial correlation matrix.
    """

    first_col = np.zeros((N,), dtype=np.complex128)
    first_col[0] = 1
    for row in range(1, N):
        dist = antenna_spacing*(row-1)
        if sigma_theta > 0 and sigma_varphi > 0:
            f = lambda delta, epsilon: np.exp(
                1j*2*np.pi*dist*np.sin(angle_varphi+delta) 
                * np.cos(angle_theta+epsilon)
            )*np.exp(
                -delta**2/(2*sigma_varphi**2)
            )/(np.sqrt(2*np.pi)*sigma_varphi) * np.exp(
                -epsilon**2/(2*sigma_theta**2)
            )/(np.sqrt(2*np.pi)*sigma_theta)
            first_col[row] = dblquad(f, -20*sigma_varphi, 20*sigma_varphi,
                                        lambda x: -20*sigma_theta,
                                        lambda x: 20*sigma_theta, args=('complex_func', True))
                                        
        elif sigma_varphi > 0:
            f = lambda delta: np.exp(
                1j*2*np.pi*dist*np.sin(angle_varphi+delta)*np.cos(angle_theta)
            )*np.exp(
                -delta**2/(2*sigma_varphi**2)
            )/(np.sqrt(2*np.pi)*sigma_varphi)
            first_col[row] = quad(f, -20*sigma_varphi, 20*sigma_varphi,
                                  complex_func=True)
        elif sigma_theta > 0:
            f = lambda epsilon: np.exp(
                1j*2*np.pi*dist*np.sin(angle_theta+epsilon)*np.cos(angle_varphi)
            )*np.exp(
                -epsilon**2/(2*sigma_theta**2)
            )/(np.sqrt(2*np.pi)*sigma_theta)
            first_col[row] = quad(f, -20*sigma_theta, 20*sigma_theta,
                                  complex_func=True)
        else:
            first_col[row] = np.exp(1j*2*np.pi*dist*np.sin(angle_varphi)*
                                       np.cos(angle_theta))
    R = toeplitz(first_col)
    R *= (N/np.trace(R))

    return R


def gen_AP_UE_statistics(L: int, N: int, K: int, sigma_varphi: float,
                         sigma_theta: float, ensemble: int) -> tuple:
    """Method to estimate some channel statistics
    
    Parameters
    ----------
    L : int
        Number of Access Points (APs).
    N : int
        Number of antennas in each AP.
    K : int
        Number of User Equipments (UEs).
    sigma_varphi : float
        Standard deviation of the nominal azimuth angle.
    sigma_theta : float
        Standard deviation of the nominal elevation angle.
    ensemble : int
        Number of setups.

    Returns
    -------
    R : np.ndarray
        Normalized covariance matrix describing the spacial correlation
        of the NLoS components.
    h : np.ndarray
        Normalized mean vector corresponding to the LoS component.
    beta : np.ndarray
        Average channel gain or large-scale fading coefficient. Modeled
        after the 3GPP model that neglect the shadow fading for each UE.
    D : np.ndarray
        Dynamic Cooperation Clustering (DCC) matrix. Element is 1 when
        the l-th AP serves the k-th UE at the n-th setup. D[l, k, n]
    K_fact : np.ndarray
        Indicates the dominance of the LoS component over the NLoS.
    los_prob : np.ndarray
        Indicates the probability of LoS according to the distance.
    """

    # Definitions:
    # We can change where these are defined later.
    sqrd_area_length = 1000  # Length of the squared area, in meters
    antenna_spacing = .5  # Measured in wavelengths
    sigma_sf_los = 4  # LoS shadow fading standard deviation
    sigma_sf_nlos = 10  # NLoS shadow fading standard deviation
    bandwidth = 20*1e6  # Bandwidth of the channel in Hz
    noise_fig = 7  # Noise figure in dB
    noise_var = -174 + 10*np.log10(bandwidth) + noise_fig  # Noise power in dBm
    alpha = 36.7  # Pathloss exponent
    constant_term = -30.5
    decorrelation_distance = 9  # Minimal distance for correlated shadow fading
    height_diff = 10  # Distance between AP and UE in the vertical axis

    beta = np.zeros((L, K, ensemble), dtype=np.complex128)
    R = np.zeros((N, N, L, K, ensemble), dtype=np.complex128)
    dist = np.zeros((L, K, ensemble), dtype=np.float64)
    D = np.zeros((L, K, ensemble), dtype=np.float64)
    master_APs = np.zeros((K,), dtype=np.float64)

    # Calculation:
    wrap_h = np.tile(np.array((-sqrd_area_length, 0, sqrd_area_length)),
                     (3, 1))
    wrap_v = wrap_h.T
    wrap_locations = (wrap_h.flatten() + 1j*wrap_v.flatten()).reshape((1, 9))
    for n in range(ensemble):
        postions_AP = (np.random.randn(L, 1)+1j*np.random.randn(L, 1)) \
            * sqrd_area_length
        # import pdb; pdb.set_trace()
        wrapped_AP_locations = np.tile(postions_AP, (1, 9)) \
            + np.tile(wrap_locations, (L, 1))
        positions_UE = np.zeros((K, 1), dtype=np.complex128)
        shadow_C = sigma_sf_los**2 * np.ones((K, K), dtype=np.float64)
        shadow_realizations_AP = np.zeros((K, L), dtype=np.float64)
        for k in range(K):  # Iterate UEs
            position_UE = (np.random.randn(1,)+1j*np.random.randn(1,)) \
                * sqrd_area_length
            dist_mat = np.abs(
                wrapped_AP_locations \
                    - np.tile(position_UE, wrapped_AP_locations.shape)
            )
            dist_APs_UE = np.min(dist_mat, axis=1)
            idx_position = np.argmin(dist_mat, axis=1)
            dist[:, k, n] = np.sqrt(height_diff**2 + dist_APs_UE**2)
            if k > 0:
                shortest_dists = np.zeros((k, 1), dtype=np.float64)
                for i in range(k):
                    shortest_dists[i] = np.min(np.abs(
                        position_UE - positions_UE[i] + wrap_locations
                    ))
                    new_col = sigma_sf_los**2 * 2**(-shortest_dists
                                                    / decorrelation_distance)
                    term_1 = new_col.T/shadow_C[:k, :k]
                    mean_values = term_1 @ shadow_realizations_AP[:k, :]
                    std_val = np.sqrt(sigma_sf_los**2 - term_1@new_col)
            else:
                mean_values = 0
                std_val = sigma_sf_los
                new_col = []
            shadowing = mean_values + std_val*np.random.randn(1, L)
            # import pdb; pdb.set_trace()
            beta[:, k, n] = constant_term - alpha*np.log10(dist[:, k, n]) \
                + shadowing.flatten() - noise_var
            shadow_C[:k, k] = new_col
            shadow_C[k, :k] = new_col
            shadow_realizations_AP[k, :] = shadowing
            positions_UE[k] = position_UE
            master_idx = np.argmax(beta[:, k, n])
            D[master_idx, k, n] = 1
            for l in range(L):
                angle_varphi = np.angle(
                    positions_UE[k]-wrapped_AP_locations[l, idx_position[l]]
                )
                angle_theta = np.arcsin(height_diff/dist[l, k, n])
                R[:, :, l, k, n] = 10**(beta[l, k, n]/10) \
                    * corr_mat_local_scatter(N, angle_varphi, angle_theta,
                                             sigma_varphi, sigma_theta,
                                             antenna_spacing)
    
    return R


def gen_channels(L: int, K: int, N: int, R: np.ndarray, ensemble: int):
    """
    Method to generate channel realizations for all UEs in the network.
    
    Parameters
    ----------
    L : int
        Number of Access Points (APs).
    K : int
        Number of User Equipments (UEs) in the network.
    N : int
        Number of antennas per AP.
    R : np.ndarray
        Array with the spacial correlation between APs and UEs,
        normalized by noise variance.
    ensemble : int
        Number of channel realizations.

    Return
    ------
    H : np.ndarray
        Channel matrix with L*N x ensemble x K, H[:, n, k] is the n-th
        collective channel realization for the k-th UE.
    """

    # Rayleigh fading channel
    H = np.random.randn(L*N, ensemble, K)+1j*np.random.randn(L*N, ensemble, K)
    for l in range(L):
        for k in range(K):
            H[l*N:(l+1)*N, :, k] = np.sqrt(.5)*sqrtm(R[:, :, l, k]) @ H[l*N:(l+1)*N, :, k]
    
    return H
    