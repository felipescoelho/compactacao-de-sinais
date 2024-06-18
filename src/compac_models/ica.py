"""ica.py

Script with class for independent component analysis.

luizfelipe.coelho@smt.ufrj.br
May 27 , 2024
"""


import numpy as np
from sklearn.decomposition import FastICA
from .base import CompressBase


class ICA(CompressBase):
    """A class for ICA model."""

    def __init__(self, **kwargs):
        """
        ICA Model
        
        Parameters
        ----------
        **kwargs : dict
            - n_variables : int
                Number of variables in the date set.
            - n_components : int
                Number of components
            - fun : str
                Name of the selected function for the FastICA
                implmentation.
        """
        super().__init__(**kwargs)
        self.n_variables = kwargs['n_variables']
        self.n_components = kwargs['n_components']
        fun_list = ['logcosh', 'exp', 'cube']
        self.fun = kwargs['fun'] if kwargs['fun'] in fun_list else 'exp'
        self.model = FastICA(n_components=kwargs['n_components'],
                             algorithm='parallel', fun=self.fun)

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def compress(self, X: np.ndarray):
        return self.model.transform(X)
    
    def decompress(self, T: np.ndarray):
        return self.model.inverse_transform(T)
