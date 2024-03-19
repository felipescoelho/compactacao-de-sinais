"""main.py

Script for "Compactação de Sinais" experiments.

luizfelipe.coelho@smt.ufrj.br
Mar 13, 2024
"""


import os
import argparse
import numpy as np


def arg_parser() -> argparse.Namespace:
    """Method to parse arguments."""
    parser = argparse.ArgumentParser()
    help_mode = 'Operation mode for the code. You can select what it will do.'
    parser.add_argument('-m', '--mode', type=str, help=help_mode,
                        choices=['gen_dataset', 'run_experiment'])
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parser()
    if args.mode == 'gen_dataset':
        os.makedirs('dataset')
