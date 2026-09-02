from pathlib import Path

path = Path("src/bochan/models/components/layers/kernel_layers.py")
text = path.read_text(encoding="utf-8")
old = '''from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch import settings
from gpytorch.models import ExactGP
'''
new = '''from gpytorch import settings
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
'''
if old not in text:
    raise SystemExit("gpytorch import block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
