"""Non-Gaussian regression models.

Directory layout:

    regression/non_gaussian/<model>/{base, deep, high_dim, robust}

Only base model packages are imported here. Optional deep / high-dimensional /
robust variants should be imported from their subpackages directly.
"""

from .poisson import *
from .beta import *
from .gamma import *
from .negative_binomial import *
from .multioutput import NonGaussianModelList
