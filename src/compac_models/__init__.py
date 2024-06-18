"""compac_models Package

Folder with code for models in signal compression.

luizfelipe.coelho@smt.ufrj.br
Abr 26, 2024
"""


__all__ = ['NLPCA', 'ICA', 'KPCA', 'PCA']


from .nlpca import NLPCA
from .ica import ICA
from .kpca import KPCA
from .pca import PCA
