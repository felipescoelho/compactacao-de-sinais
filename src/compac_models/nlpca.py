"""nlpca.py

Script with class for nonlinear principal component analysis.

luizfelipe.coelho@smt.ufrj.br
Apr 24, 2024
"""


import keras as kr
import keras_tuner as kt
from keras_tuner import HyperParameter as hp
from .base import CompressBase


class NLPCA(CompressBase):
    """"""

    @staticmethod
    def build_model(input_shape: tuple, hp):
        """"""
        input_layer = kr.Input(shape=(*input_shape,))
        hp_nonlinear_units = hp.Choice(
            name='nonlinear_units', values=[2, 4, 8, 16, 32, 64, 128, 256, 512]
        )
        hp_latent_variables = hp.Int(name='number_latent', min_value=1,
                                     max_value=input_shape[0])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        nl_h1 = kr.layers.Dense(units=hp_nonlinear_units,
                                activation='relu')(input_layer)
        lin_lat = kr.layers.Dense(units=hp_latent_variables,
                                  activation='linear')(nl_h1)
        nl_h2 = kr.layers.Dense(units=hp_nonlinear_units,
                                activation='relu')(lin_lat)
        output_layer = kr.layers.Dense(input_shape[0])(nl_h2)
        model = kr.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=kr.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=kr.losses.MeanSquaredError(reduction='sum_over_batch_size'),
            metrics=[kr.metrics.MeanSquaredError()]
        )

        return model

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_variables = kwargs['num_variables']
        self.model = self.build_model((self.num_variables,))

    def fit(self):
        """"""
        tuner = kt.RandomSearch(self.model, objective=)
