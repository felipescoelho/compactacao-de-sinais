"""nlpca.py

Script with class for nonlinear principal component analysis.

luizfelipe.coelho@smt.ufrj.br
Apr 24, 2024
"""


import keras as kr
import keras_tuner as kt
import numpy as np
from keras_tuner import HyperParameters as hp
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
from .base import CompressBase


class NLPCA(CompressBase):
    """A class for the NLPCA Model using ANN."""

    def __init__(self, **kwargs) -> None:
        """NLPCA Model
        
        Parameters
        ----------
        **kwargs : dict
            - num_variables : int
                Number of variables in the data set.
            - force_reduction : bool
                If True, the model is forced to have less latent
                variables than input variables.
        """
        super().__init__(**kwargs)
        self.num_variables = kwargs['num_variables']
        self.max_latent = self.num_variables-1 if kwargs['force_reduction'] \
            else self.num_variables
        self.model = None
        self.encoder = None
        self.decoder = None

    def fit(self, X: np.ndarray, tune_flag=False):
        """Method to fit the Associative Neural Network.
        
        Parameters
        ----------
        X : np.ndarray
            Train and validation data set as a single array. The
            dimensions are K x N, where N is the number of variables and
            K is the number of observations.
        tune_flag : bool
            If true, the model will autotune its hyperparameters.
        
        Return
        ------
        hist : dict
            A summary of the train history.
        """

        def model_build(hp):
            input_layer = kr.Input(shape=(self.num_variables,))
            hp_nonlinear_units = hp.Choice(
                name='nonlinear_units',
                values=[2, 4, 8, 16, 32, 64, 128, 256, 512]
            )
            hp_latent_variables = hp.Int(name='number_latent', min_value=1,
                                            max_value=self.max_latent)
            hp_learning_rate = hp.Choice('learning_rate',
                                         values=[1e-2, 1e-3, 1e-4])
            nl_h1 = kr.layers.Dense(units=hp_nonlinear_units,
                                    activation='sigmoid')(input_layer)
            lin_lat = kr.layers.Dense(units=hp_latent_variables,
                                      activation='linear',
                                      name='bottleneck')(nl_h1)
            nl_h2 = kr.layers.Dense(units=hp_nonlinear_units,
                                    activation='sigmoid')(lin_lat)
            output_layer = kr.layers.Dense(self.num_variables,
                                           activation='linear')(nl_h2)
            model = kr.models.Model(inputs=input_layer,
                                    outputs=output_layer)
            model.compile(
                optimizer=kr.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=kr.losses.MeanSquaredError(reduction='sum_over_batch_size')
            )

            return model
        
        # Tuning
        tuner = kt.RandomSearch(
            model_build, objective='val_loss', max_trials=10,
            max_consecutive_failed_trials=3, directory='nlpca_models',
            overwrite=tune_flag
        )
        tuner.search(X, X, batch_size=250, epochs=250, validation_split=.2,
                     callbacks=[kr.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=5,
                                                           min_delta=1e-6)],
                     shuffle=True, verbose=0)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.hypermodel.build(best_hp)
        # Train
        hist = self.model.fit(
            X, X, batch_size=250, epochs=250, validation_split=.2,
            callbacks=[kr.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5, min_delta=1e-6)],
            shuffle=True
        )
        # Split into encoder and decoder
        bottleneck_layer = self.model.get_layer(name='bottleneck')
        self.encoder = kr.models.Model(self.model.inputs,
                                       bottleneck_layer.output)
        self.decoder = kr.models.Model(bottleneck_layer.output,
                                       self.model.outputs)
        
        return hist.history

    def compress(self, X: np.ndarray):
        """Method to use model to compress a given data set.
        
        Parameters
        ----------
        X : np.ndarray
            Input data as a K x N array. (K = number of observations and
            N = number of variables)
        
        Returns
        -------
        T : np.ndarray
            Output data as a K x n array. (n = number of latent
            variables) this is the compressed signal if n < N.
        """
        return self.encoder(X)

    def decompress(self, T: np.ndarray):
        """Method to decompress a given data set.
        
        Parameters
        ----------
        T : np.ndarray
            Input array as a K x n array. (K = number of observations,
            n = number of latent variables)
        
        Returns
        -------
        X_hat : np.ndarray
            Output data as a K x N array. (N = number of variables) This
            is the reconstructed signal.
        """
        return self.decoder(T)


class PrincipalCurve:

    @staticmethod
    def project(X, p, s):
        '''
        Get interpolating s values for projection of X onto the curve
        defined by (p, s)
        @param X: data
        @param p: curve points
        @param s: curve parameterisation
        @returns: interpolating parameter values, projected points on
        curve, sum of square distances
        '''
        s_interp = np.zeros(X.shape[0])  # Interpolating Parameters
        p_interp = np.zeros(X.shape)  # Projected Points
        d_sq = 0  # Sum of distances
        
        for i in range(0, X.shape[0]):
            z = X[i, :]
            # Parallel Component
            seg_proj = (((p[1:] - p[0:-1]).T)
                        * np.einsum('ij,ij->i', z - p[0:-1], p[1:] - p[0:-1])
                        / np.power(np.linalg.norm(p[1:] - p[0:-1],
                                                  axis=1), 2)).T
            # Perpendicular Component
            proj_dist = (z - p[0:-1]) - seg_proj
            dist_endpts = np.minimum(np.linalg.norm(z - p[0:-1], axis=1),
                                     np.linalg.norm(z - p[1:], axis=1))
            dist_seg = np.maximum(np.linalg.norm(proj_dist, axis=1),
                                  dist_endpts)
            idx_min = np.argmin(dist_seg)
            q = seg_proj[idx_min]
            s_interp[i] = (np.linalg.norm(q)
                           / np.linalg.norm(p[idx_min + 1, :]-p[idx_min, :])) \
                * (s[idx_min+1]-s[idx_min]) + s[idx_min]
            p_interp[i] = (s_interp[i] - s[idx_min]) \
                * (p[idx_min+1, :] - p[idx_min, :]) + p[idx_min, :]
            d_sq += np.linalg.norm(proj_dist[idx_min])**2

        return s_interp, p_interp, d_sq
    
    @staticmethod
    def renorm_parameterisation(p):
        '''
        Renormalise curve to unit speed 
        @param p: curve points
        @returns: new parameterisation
        '''
        seg_lens = np.linalg.norm(p[1:] - p[0:-1], axis = 1)
        s = np.zeros(p.shape[0])
        s[1:] = np.cumsum(seg_lens)
        s = s/sum(seg_lens)

        return s
    
    def __init__(self, k = 3):
        self.k = k
        self.p = None
        self.s = None
        self.p_interp = None
        self.s_interp = None

    def fit(self, X, p = None, w = None, max_iter = 10, tol = 1e-3):
        '''
        Fit principal curve to data
        @param X: data
        @param p: starting curve (optional)
        @param w: data weights (optional)
        @param max_iter: maximum number of iterations 
        @param tol: tolerance for stopping condition
        @returns: None
        '''
        # Starting point PCA 1st component:
        pca = PCA(n_components = X.shape[1])
        pca.fit(X)
        pc1 = pca.components_[:, 0]
        if p is None:
            p = np.kron(np.dot(X, pc1)/np.dot(pc1, pc1), pc1).reshape(X.shape)
            order = np.argsort([np.linalg.norm(p[0, :] - p[i, :]) 
                                for i in range(0, p.shape[0])])
            p = p[order]
        s = self.renorm_parameterisation(p)
        # Memory allocation:
        p_interp = np.zeros(X.shape)
        s_interp = np.zeros(X.shape[0])
        # Computation:
        d_sq_old = np.Inf
        for it in range(0, max_iter):
            # Calculate distances
            print(it)
            s_interp, p_interp, d_sq = self.project(X, p, s)
            print(np.abs(d_sq - d_sq_old))
            if np.abs(d_sq - d_sq_old) < tol:
                break
            d_sq_old = d_sq
            order = np.argsort(s_interp)
            # Family of curves
            spline = [
                UnivariateSpline(s_interp[order], X[order, j], k=self.k, w=w)
                for j in range(0, X.shape[1])
            ]
            # Gen projections by curves
            p = np.zeros((len(s_interp), X.shape[1]))
            for j in range(0, X.shape[1]):
                p[:, j] = spline[j](s_interp[order])
            idx = [i for i in range(0, p.shape[0]-1) if (p[i] != p[i+1]).any()]
            p = p[idx, :]
            s = self.renorm_parameterisation(p)
            
        self.s = s
        self.p = p
        self.p_interp = p_interp
        self.s_interp = s_interp
