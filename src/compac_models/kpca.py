"""kpca.py

Script with class for Kernel Principal Component Analysis.

luizfelipe.coelho@smt.ufrj.br
Jun 18, 2024
"""


import numpy as np
from sklearn.decomposition import KernelPCA
from .base import CompressBase


class KPCA(CompressBase):
    """A Class for KPCA model."""

    def __init__(self, **kwargs):
        """
        KPCA Model
        
        Parameters
        ----------
        **kwargs : dict
            - n_variables : int
                Number of variables in the data set
            - n_components : int
                Number of components (latent variables)
        """
        super().__init__(**kwargs)
        self.n_variables = kwargs['n_variables']
        self.n_components = kwargs['n_components']
        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine',
                       'precomputed']
        self.kernel = kwargs['kernel'] if kwargs['kernel'] in kernel_list \
            else 'rbf'
        self.model = KernelPCA(n_components=self.n_components,
                               kernel=self.kernel, gamma=10,
                               fit_inverse_transform=True, alpha=0.1)
    
    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def compress(self, X: np.ndarray):
        return self.model.transform(X)

    def decompress(self, T: np.ndarray):
        return self.model.inverse_transform(T)