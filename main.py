"""main.py

Script for "Compactação de Sinais" experiments.

luizfelipe.coelho@smt.ufrj.br
Mar 13, 2024
"""


import os
import argparse
import pywt
import numpy as np
from src.channel import gen_channels, gen_AP_UE_statistics
from src.utils import dft_channel, idft_channel, calculate_gzf, calculate_bps
from src.compac_models import PCA, ICA, KPCA, NLPCA



def arg_parser() -> argparse.Namespace:
    """Method to parse arguments."""
    parser = argparse.ArgumentParser()
    help_mode = 'Operation mode for the code. You can select what it will do.'
    parser.add_argument('-m', '--mode', type=str, help=help_mode,
                        choices=['gen_dataset', 'train_models',
                                 'run_sim'])
    parser.add_argument('-L', '--num_ap', type=int,
                        help='Number of Access Points (AP)', default=25)
    parser.add_argument('-N', '--num_antennas', type=int,
                        help='Number of antennas in each AP', default=16)
    parser.add_argument('-K', '--num_ue', type=int,
                        help='Number of User Equipments', default=14)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parser()
    match args.mode:
        case 'gen_dataset':
            # Definitions:
            R, D = gen_AP_UE_statistics(args.num_ap, args.num_antennas, args.num_ue,
                                        5, 5, 1)
            np.save('correlation2_mat.npy', R)
            np.save('position2_mat.npy', D)
        case 'train_models':
            rng = np.random.default_rng()
            with open('correlation2_mat.npy', 'rb') as f:
                R = np.load(f)
            with open('position2_mat.npy', 'rb') as f:
                dist = np.load(f)
            N, _, L, K = R.shape
            H_train = gen_channels(R, 10000, rng.integers(9999999999))
            H_train_dft = dft_channel(H_train, N)

            # Generate models:
            pca1_5_list = [PCA(n_variables=N*K, n_components=5)]
            pca1_5_list = [PCA(n_variables=N*K, n_components=5)]
            pca1_5_list = [PCA(n_variables=N*K, n_components=5)]

            H_train_idft = idft_channel(H_train_dft, N)
            print(dist < 50)
            
        
        case 'run_sim':
            rng = np.random.default_rng()
            with open('correlation2_mat.npy', 'rb') as f:
                R = np.load(f)
            with open('position2_mat.npy', 'rb') as f:
                dist = np.load(f)
            N, _, L, K = R.shape
            H_test = gen_channels(R, 1, rng.integers(9999999999))
            W = calculate_gzf(H_test)
            A = (dist < 50).astype(np.int32)
            sigma_n2 = (10**-9.2)/1000
            sinr, r = calculate_bps(H_test, W, A, sigma_n2)
            print(np.sum(r))