"""pca.py

Script with class for principal component analysis.

luizfelipe.coelho@smt.ufrj.br
Ju 18, 2024
"""


import numpy as np
from sklearn.decomposition import PCA as pca
from .base import CompressBase


class PCA(CompressBase):
    """A Class for PCA model."""

    def __init__(self, **kwargs):
        """
        PCA Model
        
        Parameters
        ----------
        **kwargs : dict
            - n_variables : int
                Number of variables in the data set.
            - n_components : int
                Number of components (latent variables)
        """
        super().__init__(**kwargs)
        self.n_variables = kwargs['n_variables']
        self.n_components = kwargs['n_components']
        self.model = pca(n_components=self.n_components)

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def compress(self, X: np.ndarray):
        return self.model.transform(X)

    def decompress(self, T: np.ndarray):
        return self.model.inverse_transform(T)
    