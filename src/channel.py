"""channel.py

Script to simulate the channel matrix of cell-free massive MIMO
communication system.

luizfelipe.coelho@smt.ufrj.br
Mar 20, 2024
"""


import numpy as np
from scipy.integrate import quad, dblquad, quad_vec
from scipy.linalg import toeplitz, sqrtm
from multiprocessing import cpu_count, Pool
from tqdm import tqdm


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


def real_fun(epsilon, delta, dist, angle_varphi, angle_theta, sigma_varphi,
                sigma_theta):
    """"""
    return (np.exp(
        1j*2*np.pi*dist*np.sin(angle_varphi+delta)*np.cos(angle_theta+epsilon)
    )*np.exp(
        -delta**2/(2*sigma_varphi**2)
    )/(np.sqrt(2*np.pi)*sigma_varphi)*np.exp(
        -epsilon**2/(2*sigma_theta**2)
    )/(np.sqrt(2*np.pi)*sigma_theta)).real


def quick_int_double_var_real(dist, angle_varphi, angle_theta, sigma_varphi,
                              sigma_theta):
    """"""
    args = (dist, angle_varphi, angle_theta, sigma_varphi, sigma_theta)

    return quad_vec(quick_int_single_var_real, -20*sigma_theta, 20*sigma_theta,
                    args=(*args,))


def quick_int_single_var_real(delta, dist, angle_varphi, angle_theta,
                              sigma_varphi, sigma_theta):
    """"""
    args = (delta, dist, angle_varphi, angle_theta, sigma_varphi, sigma_theta)

    return quad_vec(real_fun, -20*sigma_varphi, 20*sigma_varphi, args=(*args,))


def imaginary_fun(epsilon, delta, dist, angle_varphi, angle_theta,
                  sigma_varphi, sigma_theta):
    """"""
    return (np.exp(
        1j*2*np.pi*dist*np.sin(angle_varphi+delta)*np.cos(angle_theta+epsilon)
    )*np.exp(
        -delta**2/(2*sigma_varphi**2)
    )/(np.sqrt(2*np.pi)*sigma_varphi)*np.exp(
        -epsilon**2/(2*sigma_theta**2)
    )/(np.sqrt(2*np.pi)*sigma_theta)).imag


def quick_int_double_var_imag(dist, angle_varphi, angle_theta, sigma_varphi,
                              sigma_theta):
    """"""
    args = (dist, angle_varphi, angle_theta, sigma_varphi, sigma_theta)

    return quad_vec(quick_int_single_var_imag, -20*sigma_theta, 20*sigma_theta,
                    args=(*args,), workers=-1)


def quick_int_single_var_imag(delta, dist, angle_varphi, angle_theta,
                              sigma_varphi, sigma_theta):
    """"""
    args = (delta, dist, angle_varphi, angle_theta, sigma_varphi, sigma_theta)

    return quad_vec(imaginary_fun, -20*sigma_varphi, 20*sigma_varphi,
                    args=(*args,), workers=-1)


def corr_mat_local_scatter(N: int, angle_varphi: float, angle_theta: float,
                           sigma_varphi: float, sigma_theta: float,
                           antenna_spacing: float) -> np.ndarray:
    """
    Method to generate the spacial correlation matrix for a given local
    scattering model for Gaussian angular distribution and ULA.

    Equation (2.18) from cell-free-book

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

    col = np.zeros((N,), dtype=np.complex128)
    col[0] = 1
    for n in range(1, N):
        dist = antenna_spacing*(n-1)
        if sigma_theta > 0 and sigma_varphi > 0:
            f_real = lambda delta, epsilon: (np.exp(
                1j*2*np.pi*dist*np.sin(angle_varphi+delta)
                * np.cos(angle_theta+epsilon)
            )*np.exp(
                -delta**2/(2*sigma_varphi**2)
            )/(np.sqrt(2*np.pi)*sigma_varphi) * np.exp(
                -epsilon**2/(2*sigma_theta**2)
            )/(np.sqrt(2*np.pi)*sigma_theta)).real
            f_img = lambda delta, epsilon: (np.exp(
                1j*2*np.pi*dist*np.sin(angle_varphi+delta)
                * np.cos(angle_theta+epsilon)
            )*np.exp(
                -delta**2/(2*sigma_varphi**2)
            )/(np.sqrt(2*np.pi)*sigma_varphi) * np.exp(
                -epsilon**2/(2*sigma_theta**2)
            )/(np.sqrt(2*np.pi)*sigma_theta)).imag
            r_comp = quad_vec(
                lambda delta: quad_vec(
                    lambda epsilon: f_real(delta, epsilon), -20*sigma_varphi,
                    20*sigma_varphi
                )[0], -20*sigma_theta, 20*sigma_theta
            )[0]
            i_comp = quad_vec(
                lambda delta: quad_vec(
                    lambda epsilon: f_img(delta, epsilon), -20*sigma_varphi,
                    20*sigma_varphi
                )[0], -20*sigma_theta, 20*sigma_theta
            )[0]
            # r_comp, _ = dblquad(f_real, -sigma_varphi, sigma_varphi,
            #                     lambda x: -sigma_theta, lambda x: sigma_theta)
            # i_comp, _ = dblquad(f_img, -sigma_varphi, sigma_varphi,
            #                     lambda x: -sigma_theta, lambda x: sigma_theta)
            col[n] =  r_comp + 1j*i_comp
            # col[n] = quick_int_double_var_real(dist, angle_varphi, angle_theta,
            #                                    sigma_varphi, sigma_theta) \
            #     + 1j*quick_int_double_var_imag(dist, angle_varphi, angle_theta,
            #                                    sigma_varphi, sigma_theta)
        elif sigma_varphi > 0:
            f = lambda delta: np.exp(
                1j*2*np.pi*dist*np.sin(angle_varphi+delta)*np.cos(angle_theta)
            )*np.exp(
                -delta**2/(2*sigma_varphi**2)
            )/(np.sqrt(2*np.pi)*sigma_varphi)
            col[n] = quad(f, -sigma_varphi,
                                  sigma_varphi, complex_func=True)
        elif sigma_theta > 0:
            f = lambda epsilon: np.exp(
                1j*2*np.pi*dist*np.sin(angle_theta+epsilon)*np.cos(angle_varphi)
            )*np.exp(
                -epsilon**2/(2*sigma_theta**2)
            )/(np.sqrt(2*np.pi)*sigma_theta)
            col[n] = quad(f, -sigma_theta, sigma_theta,
                                  complex_func=True)
        else:
            col[n] = np.exp(1j*2*np.pi*dist*np.sin(angle_varphi)*
                                    np.cos(angle_theta))
    R = toeplitz(col)
    R *= N/np.trace(R)  # Normalization

    return R


def multiprocessing_worker(data_in):
    """"""
    l, N, sigma_varphi, sigma_theta, antenna_spacing, angle_varphi, \
        angle_theta, beta_lk = data_in
    R = beta_lk*corr_mat_local_scatter(N, angle_varphi, angle_theta,
                                       sigma_varphi, sigma_theta,
                                       antenna_spacing)
    
    return (l, R)


def single_run(data: tuple):
    """Method to generate a single realization of the setup."""

    # Get stup data
    setup_info, AP_info, seed = data
    area_len, antenna_spacing, sigma_sf_los, noise_var, alpha, \
        constant_term, decorr_distance, height, N, L, K = setup_info
    wrapped_AP_locations, wrap_locations, sigma_varphi, sigma_theta = AP_info
    # Random number generator
    rng = np.random.default_rng(seed=seed)

    # Memory allocation for UEs
    UEs_position = np.zeros((K, 1), dtype=np.complex128)
    R_shadow = sigma_sf_los**2 * np.ones((K, K), dtype=np.float64)
    shadow_realizations_AP = np.zeros((K, L), dtype=np.float64)
    beta = np.zeros((L, K), dtype=np.complex128)
    dist = np.zeros((L, K), dtype=np.float64)
    R = np.zeros((N, N, L, K), dtype=np.complex128)
    for k in tqdm(range(K)):
        UE_position = area_len * (rng.standard_normal((1,))
                                  + 1j*rng.standard_normal((1,)))
        dist_mat = np.abs(wrapped_AP_locations - UE_position)
        dist_APs_UE = np.min(dist_mat, axis=1)
        idx_position = np.argmin(dist_mat, axis=1)
        dist[:, k] = np.sqrt(height**2 + dist_APs_UE**2)
        if k > 0:
            # See: S. Kay "Fundamental of Statistical Signal Processing:
            # Estimation Theory" -- Theorem 10.2 on shadow fading
            # realizations when previous UEs shadow fading realization have
            # already been computed.
            shortest_dists = np.zeros((k,), dtype=np.float64)
            for i in range(k):
                shortest_dists[i] = np.min(np.abs(
                    UE_position - UEs_position[i] + wrap_locations
                ))
            new_col = sigma_sf_los**2*2**(-shortest_dists/decorr_distance)
            spam = new_col.T @ np.linalg.pinv(R_shadow[:k, :k])
            mean_values = spam @ shadow_realizations_AP[:k, :]
            std_val = np.sqrt(sigma_sf_los**2 - np.dot(spam, new_col))
            R_shadow[:k, k] = new_col
            R_shadow[k, :k] = new_col
            shadowing = mean_values + std_val*rng.standard_normal((L,))
        else:
            shadowing = sigma_sf_los*rng.standard_normal((L,))
        beta[:, k] = constant_term - alpha*np.log10(dist[:, k]) + shadowing \
            - noise_var
        shadow_realizations_AP[k, :] = shadowing
        UEs_position[k] = UE_position
        data_in = [
            (l, N, sigma_varphi, sigma_theta, antenna_spacing,
             np.angle(UEs_position[k]-wrapped_AP_locations[l, idx_position[l]]),
             np.arcsin(height/dist[l, k]), 10**(beta[l, k]/10))
            for l in range(L)
        ]
        with Pool(cpu_count()) as pool:
            result_list = pool.map(multiprocessing_worker, data_in)
        for idx in range(L):
            l, R_lk = result_list[idx]
            R[:, :, l, k] = R_lk
        # for l in range(L):
        #     angle_varphi = np.angle(
        #         UEs_position[k]-wrapped_AP_locations[l, idx_position[l]]
        #     )
        #     angle_theta = np.arcsin(height/dist[l, k])
        #     R[:, :, l, k] = 10**(beta[l, k]/10) * corr_mat_local_scatter(
        #         N, angle_varphi, angle_theta, sigma_varphi, sigma_theta,
        #         antenna_spacing
        #     )
    
    return R, dist


def gen_AP_UE_statistics(L: int, N: int, K: int, sigma_varphi: float,
                         sigma_theta: float, ensemble: int, seed=42) -> tuple:
    """Method to estimate some channel statistics based of code from
    cell-free-book.
    
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
    seed : int
        Seed number for RNG.

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
    area_len = 100  # Length of the squared area, in meters
    antenna_spacing = .5  # Measured in wavelengths
    sigma_sf_los = 4  # LoS shadow fading standard deviation
    sigma_sf_nlos = 10  # NLoS shadow fading standard deviation
    bandwidth = 1900*1e6  # Bandwidth of the channel in Hz
    noise_fig = 7  # Noise figure in dB
    noise_var = -174 + 10*np.log10(bandwidth) + noise_fig  # Noise power in dBm
    alpha = 36.7  # Pathloss exponent
    constant_term = -30.5
    decorr_distance = 9  # Minimal distance for correlated shadow fading
    height = 10  # Distance between AP and UE in the vertical axis
    
    # Calculation:
    spam = np.tile(np.array((-area_len, 0, area_len)), (3, 1))
    wrap_locations = spam.T.flatten() + 1j*spam.flatten()
    rng = np.random.default_rng(seed=seed)
    # Randomized position for APs
    APs_position = area_len * (rng.standard_normal((L,))
                               + 1j*rng.standard_normal((L,)))
    wrapped_AP_locations = np.tile(APs_position, (9, 1)).T \
        + np.tile(wrap_locations, (L, 1))
    setup_info = (area_len, antenna_spacing, sigma_sf_los, noise_var, alpha,
                  constant_term, decorr_distance, height, N, L, K)
    AP_info = (wrapped_AP_locations, wrap_locations, sigma_varphi, sigma_theta)
    data_list = [(setup_info, AP_info, rng.integers(99999999))
                 for _ in range(ensemble)]
    if ensemble > 1:
        with Pool(cpu_count()) as pool:
            results = pool.map(single_run, data_list)
        
        R = np.zeros((N, N, L, K, ensemble), dtype=np.complex128)
        dist = np.zeros((L, K, ensemble), dtype=np.complex128)
        for it in range(ensemble):
            R[:, :, :, :, it], dist[:, :, it] = results[it]
    else:
        R, dist = single_run(*data_list)
    
    return R, dist


def gen_channels(L: int, K: int, N: int, R: np.ndarray, ensemble: int,
                 seed=42):
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
    rng = np.random.default_rng(seed=seed)
    H = rng.standard_normal((L*N, ensemble, K)) \
        + 1j*rng.standard_normal((L*N, ensemble, K))
    
    for it in range(ensemble):
        for l in range(L):
            for k in range(K):
                H[l*N:(l+1)*N, it, k] = np.sqrt(.5)*sqrtm(R[:, :, l, k, it]) \
                    @ H[l*N:(l+1)*N, it, k]
    
    return H
    