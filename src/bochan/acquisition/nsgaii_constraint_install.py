"""Install the objective-aware NSGA-II wrapper on the public optimizer package."""

from __future__ import annotations


_INSTALLED = False


def install_nsgaii_constraint_compat() -> None:
    """Route public NSGA-II calls through objective-space constraints."""

    global _INSTALLED
    if _INSTALLED:
        return

    import bochan.optim as optim_package
    from bochan.optim.nsgaii_constraint_compat import optimize_acqf_nsgaii

    optim_package.optimize_acqf_nsgaii = optimize_acqf_nsgaii
    _INSTALLED = True


__all__ = ["install_nsgaii_constraint_compat"]
