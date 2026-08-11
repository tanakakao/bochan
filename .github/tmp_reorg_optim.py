from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path('.')
OPTIM = ROOT / 'src/bochan/optim'

MOVES = {
    'standard.py': 'gradient/botorch.py',
    'torch_opt.py': 'gradient/torch.py',
    'torch_multitask.py': 'gradient/multitask.py',
    'evo.py': 'evolutionary/core.py',
    'nsgaii.py': 'nsgaii/core.py',
    'nsgaii_adapter.py': 'nsgaii/adapter.py',
    'nsgaii_constraints.py': 'nsgaii/constraints.py',
    'nsgaii_diversity.py': 'nsgaii/diversity.py',
    'nsgaii_outputs.py': 'nsgaii/outputs.py',
    'nsgaii_strategy.py': 'nsgaii/strategy.py',
    'thompson_sampling.py': 'thompson/core.py',
    'thompson_sampling_adapter.py': 'thompson/adapter.py',
    'llm.py': 'llm/candidate_set.py',
}

for src_rel, dst_rel in MOVES.items():
    src = OPTIM / src_rel
    dst = OPTIM / dst_rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'mv', str(src), str(dst)], check=True)

replacements_by_file = {
    OPTIM / 'gradient/botorch.py': {
        'from ..constraints.k_sparse import (': 'from ...constraints.k_sparse import (',
    },
    OPTIM / 'gradient/torch.py': {
        'from ..constraints.k_sparse import (': 'from ...constraints.k_sparse import (',
    },
    OPTIM / 'gradient/multitask.py': {
        'from .torch_opt import InequalitySense, LinearConstraint, TorchOptimizerName':
            'from .torch import InequalitySense, LinearConstraint, TorchOptimizerName',
        'from .torch_opt import optimize_acqf_torch as _optimize_acqf_torch':
            'from .torch import optimize_acqf_torch as _optimize_acqf_torch',
    },
    OPTIM / 'evolutionary/core.py': {
        'from ..constraints.k_sparse import (': 'from ...constraints.k_sparse import (',
    },
    OPTIM / 'nsgaii/adapter.py': {
        'from . import nsgaii as _base': 'from . import core as _base',
        'from .nsgaii_diversity import select_diverse_nsgaii_candidates':
            'from .diversity import select_diverse_nsgaii_candidates',
        'from .nsgaii_outputs import adapt_nsgaii_outputs':
            'from .outputs import adapt_nsgaii_outputs',
    },
    OPTIM / 'nsgaii/constraints.py': {
        'from .nsgaii_adapter import LinearConstraint': 'from .adapter import LinearConstraint',
        'from .nsgaii_adapter import optimize_acqf_nsgaii as _base_optimize_acqf_nsgaii':
            'from .adapter import optimize_acqf_nsgaii as _base_optimize_acqf_nsgaii',
    },
    OPTIM / 'thompson/adapter.py': {
        'from . import thompson_sampling as _base': 'from . import core as _base',
    },
}

for path, replacements in replacements_by_file.items():
    text = path.read_text(encoding='utf-8')
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f'Missing expected text in {path}: {old}')
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')

package_files = {
    OPTIM / 'gradient/__init__.py': '''"""Gradient-based acquisition optimization backends."""\n\nfrom .botorch import optimize_acqf_k_sparse, optimize_acqf_mixed_k_sparse\nfrom .multitask import optimize_acqf_torch\nfrom .torch import (\n    optimize_acqf_torch_k_sparse,\n    optimize_acqf_torch_mixed,\n    optimize_acqf_torch_mixed_k_sparse,\n)\n\n__all__ = [\n    "optimize_acqf_k_sparse",\n    "optimize_acqf_mixed_k_sparse",\n    "optimize_acqf_torch",\n    "optimize_acqf_torch_mixed",\n    "optimize_acqf_torch_k_sparse",\n    "optimize_acqf_torch_mixed_k_sparse",\n]\n''',
    OPTIM / 'evolutionary/__init__.py': '''"""Population and derivative-free acquisition optimization backends."""\n\nfrom .core import (\n    candidate_transform_mixed_factory,\n    optimize_acqf_evo,\n    optimize_acqf_evo_k_sparse,\n    optimize_acqf_evo_mixed,\n    optimize_acqf_evo_mixed_k_sparse,\n)\n\n__all__ = [\n    "candidate_transform_mixed_factory",\n    "optimize_acqf_evo",\n    "optimize_acqf_evo_k_sparse",\n    "optimize_acqf_evo_mixed",\n    "optimize_acqf_evo_mixed_k_sparse",\n]\n''',
    OPTIM / 'nsgaii/__init__.py': '''"""NSGA-II optimization backend and support utilities."""\n\nfrom .adapter import (\n    equality_constraints_to_inequality_constraints,\n    optimize_acqf_nsgaii,\n    validate_discrete_choices,\n)\nfrom .strategy import NSGAIIStrategy, build_nsgaii_strategy\n\n__all__ = [\n    "NSGAIIStrategy",\n    "build_nsgaii_strategy",\n    "equality_constraints_to_inequality_constraints",\n    "optimize_acqf_nsgaii",\n    "validate_discrete_choices",\n]\n''',
    OPTIM / 'thompson/__init__.py': '''"""Finite-pool Thompson sampling optimization backend."""\n\nfrom .adapter import optimize_thompson_sampling, optimize_thompson_sampling_mixed\n\n__all__ = [\n    "optimize_thompson_sampling",\n    "optimize_thompson_sampling_mixed",\n]\n''',
    OPTIM / 'llm/__init__.py': '''"""LLM-generated candidate-set optimization backend."""\n\nfrom .candidate_set import optimize_acqf_llm_candidate_set\n\n__all__ = ["optimize_acqf_llm_candidate_set"]\n''',
}
for path, content in package_files.items():
    path.write_text(content, encoding='utf-8')

(OPTIM / '__init__.py').write_text(
    '''"""Acquisition optimization backends.\n\nAlgorithm-specific implementations live in subpackages. This module exposes the\nstable high-level optimizer surface used by the API dispatch layer.\n"""\n\nfrom .evolutionary import (\n    candidate_transform_mixed_factory,\n    optimize_acqf_evo,\n    optimize_acqf_evo_k_sparse,\n    optimize_acqf_evo_mixed,\n    optimize_acqf_evo_mixed_k_sparse,\n)\nfrom .gradient import (\n    optimize_acqf_k_sparse,\n    optimize_acqf_mixed_k_sparse,\n    optimize_acqf_torch,\n    optimize_acqf_torch_k_sparse,\n    optimize_acqf_torch_mixed,\n    optimize_acqf_torch_mixed_k_sparse,\n)\nfrom .llm import optimize_acqf_llm_candidate_set\nfrom .nsgaii import (\n    equality_constraints_to_inequality_constraints,\n    optimize_acqf_nsgaii,\n    validate_discrete_choices,\n)\nfrom .thompson import optimize_thompson_sampling, optimize_thompson_sampling_mixed\n\n__all__ = [\n    "optimize_acqf_k_sparse",\n    "optimize_acqf_mixed_k_sparse",\n    "candidate_transform_mixed_factory",\n    "optimize_acqf_evo",\n    "optimize_acqf_evo_k_sparse",\n    "optimize_acqf_evo_mixed",\n    "optimize_acqf_evo_mixed_k_sparse",\n    "optimize_acqf_llm_candidate_set",\n    "equality_constraints_to_inequality_constraints",\n    "optimize_acqf_nsgaii",\n    "validate_discrete_choices",\n    "optimize_thompson_sampling",\n    "optimize_thompson_sampling_mixed",\n    "optimize_acqf_torch",\n    "optimize_acqf_torch_mixed",\n    "optimize_acqf_torch_k_sparse",\n    "optimize_acqf_torch_mixed_k_sparse",\n]\n''',
    encoding='utf-8',
)

generic_replacements = [
    ('bochan.optim.nsgaii_strategy', 'bochan.optim.nsgaii.strategy'),
    ('bochan.optim.nsgaii_adapter', 'bochan.optim.nsgaii.adapter'),
    ('bochan.optim.nsgaii_constraints', 'bochan.optim.nsgaii.constraints'),
    ('bochan.optim.nsgaii_diversity', 'bochan.optim.nsgaii.diversity'),
    ('bochan.optim.nsgaii_outputs', 'bochan.optim.nsgaii.outputs'),
    ('bochan.optim.thompson_sampling_adapter', 'bochan.optim.thompson.adapter'),
    ('bochan.optim.thompson_sampling', 'bochan.optim.thompson.core'),
    ('bochan.optim.torch_multitask', 'bochan.optim.gradient.multitask'),
    ('bochan.optim.torch_opt', 'bochan.optim.gradient.torch'),
    ('bochan.optim.standard', 'bochan.optim.gradient.botorch'),
    ('bochan.optim.evo', 'bochan.optim.evolutionary'),
]
for path in ROOT.rglob('*'):
    if not path.is_file() or path.suffix not in {'.py', '.md', '.rst', '.yml', '.yaml', '.toml'}:
        continue
    if '.git' in path.parts or path.name.startswith('tmp_optim_package_reorg') or path.name == 'tmp_reorg_optim.py':
        continue
    text = path.read_text(encoding='utf-8')
    new_text = text
    for old, new in generic_replacements:
        new_text = new_text.replace(old, new)
    if new_text != text:
        path.write_text(new_text, encoding='utf-8')

(OPTIM / 'ARCHITECTURE.md').write_text(
    '''# Optimizer package architecture\n\n`bochan.optim` is organized by optimization algorithm family rather than by\nindividual historical modules. The top-level package only aggregates the public\noptimizer surface used by API dispatch.\n\n## Layout\n\n- `gradient/`: gradient-based candidate optimization. `botorch.py` contains\n  BoTorch wrappers, `torch.py` contains `torch.optim` implementations, and\n  `multitask.py` contains the correlated-multitask restart strategy.\n- `evolutionary/`: derivative-free population/search methods such as GA, PSO,\n  simulated annealing, and CMA-ES.\n- `nsgaii/`: NSGA-II backend, strategy, API adapter, objective-space constraint\n  adaptation, output-shape normalization, and final-batch diversity selection.\n- `thompson/`: finite-pool Thompson sampling core and the high-level objective /\n  constraint adapter.\n- `llm/`: LLM-generated candidate-set backends. The acquisition function remains\n  responsible for final scoring.\n\n## Extension rule\n\nAdd a new optimizer family as a new subpackage when it has its own optimization\nsemantics or support utilities. Keep shared problem constraints under\n`bochan.constraints`; optimizer packages should only contain optimizer-facing\nadaptation. Avoid runtime monkey patches and forwarding modules for removed paths.\n''',
    encoding='utf-8',
)

(ROOT / 'tests/test_optim_package_layout.py').write_text(
    '''from __future__ import annotations\n\nfrom pathlib import Path\n\nimport bochan.optim as optim\nfrom bochan.optim import evolutionary, gradient, llm, nsgaii, thompson\n\n\ndef test_optim_flat_modules_are_removed() -> None:\n    root = Path(optim.__file__).resolve().parent\n    removed = [\n        "evo.py",\n        "standard.py",\n        "torch_opt.py",\n        "torch_multitask.py",\n        "nsgaii.py",\n        "nsgaii_adapter.py",\n        "nsgaii_constraints.py",\n        "nsgaii_diversity.py",\n        "nsgaii_outputs.py",\n        "nsgaii_strategy.py",\n        "thompson_sampling.py",\n        "thompson_sampling_adapter.py",\n        "llm.py",\n    ]\n    assert all(not (root / name).exists() for name in removed)\n\n\ndef test_optim_public_exports_use_algorithm_packages() -> None:\n    assert optim.optimize_acqf_k_sparse is gradient.optimize_acqf_k_sparse\n    assert optim.optimize_acqf_torch is gradient.optimize_acqf_torch\n    assert optim.optimize_acqf_evo is evolutionary.optimize_acqf_evo\n    assert optim.optimize_acqf_nsgaii is nsgaii.optimize_acqf_nsgaii\n    assert optim.optimize_thompson_sampling is thompson.optimize_thompson_sampling\n    assert optim.optimize_acqf_llm_candidate_set is llm.optimize_acqf_llm_candidate_set\n    assert nsgaii.build_nsgaii_strategy is not None\n''',
    encoding='utf-8',
)