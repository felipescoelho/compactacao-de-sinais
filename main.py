"""main.py

Script for "Compactação de Sinais" experiments.

luizfelipe.coelho@smt.ufrj.br
Mar 13, 2024
"""


import os
import argparse
import numpy as np
import pandas as pd
from src.channel import gen_channels, gen_AP_UE_statistics


def arg_parser() -> argparse.Namespace:
    """Method to parse arguments."""
    parser = argparse.ArgumentParser()
    help_mode = 'Operation mode for the code. You can select what it will do.'
    parser.add_argument('-m', '--mode', type=str, help=help_mode,
                        choices=['gen_dataset', 'run_experiment'])
    parser.add_argument('-L', '--num_ap', type=int,
                        help='Number of Access Points (AP)', default=25)
    parser.add_argument('-N', '--num_antennas', type=int,
                        help='Number of antennas in each AP', default=20)
    parser.add_argument('-K', '--num_ue', type=int,
                        help='Number of User Equipments', default=16)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parser()
    if args.mode == 'gen_dataset':
        # Definitions:
        
        R = gen_AP_UE_statistics(args.num_ap, args.num_antennas, args.num_ue, 5, 5, 10)
        R = np.zeros((args.num_antennas, args.num_antennas, args.num_ap, args.num_ue))
        for l in range(args.num_ap):
            for k in range(args.num_ue):
                R[:, :, l, k] = np.eye(args.num_antennas)
        H = gen_channels(args.num_ap, args.num_ue, args.num_antennas, R, 10)
        print(H.shape)