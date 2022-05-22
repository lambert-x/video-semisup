from .base import BaseWeightedLoss
from .binary_logistic_regression_loss import BinaryLogisticRegressionLoss
from .bmn_loss import BMNLoss
from .cross_entropy_loss import BCELossWithLogits, CrossEntropyLoss
from .hvu_loss import HVULoss
from .nll_loss import NLLLoss
from .ohem_hinge_loss import OHEMHingeLoss
from .ssn_loss import SSNLoss
from .cosine_simi_loss import CosineSimiLoss
from .simclr_loss_ptv_new import SimCLRLoss_PTV_New
__all__ = [
    'BaseWeightedLoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELossWithLogits',
    'BinaryLogisticRegressionLoss', 'BMNLoss', 'OHEMHingeLoss', 'SSNLoss',
    'HVULoss',
    # custom imports
    'NTXentLoss', 'SimCLRLoss', 'SimsiamLoss', 'CosineSimiLoss', 'SimCLRLoss_PTV',
    'SimCLRLoss_PTV_New'
]
