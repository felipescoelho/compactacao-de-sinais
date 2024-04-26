"""base.py

Script with base class for our compression models.

luizfelipe.coelho@smt.ufrj.br
Apr 19, 2024
"""


import numpy as np
from abc import ABC, abstractmethod


class CompressBase(ABC):
    """"""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X: np.ndarray):
        """Method to fit model using a train data set.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix containing K observation rows and N variable
            columns.
        """

    @abstractmethod
    def compress(self, X: np.ndarray):
        """Method to compress data set using trained model.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix containing K observation rows and N variable
            columns.

        Returns
        -------
        Y : np.ndarray
            Data matrix containing K observation rows and n latent
            variables.
        """

    @abstractmethod
    def decompress(self, T:np.ndarray):
        """Method to decompress data set using trained model.
        
        Parameters
        ----------
        T : np.ndarray
            Compressed data set. The dimensions will depend on the
            selected model.
        """
