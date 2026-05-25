"""Non-Gaussian regression models.

Directory layout:

    regression/non_gaussian/<model>/{base, deep, high_dim, robust}
"""

from .poisson import *
from .beta import *
from .gamma import *
from .negative_binomial import *
