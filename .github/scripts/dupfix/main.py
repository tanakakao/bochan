from __future__ import annotations

from .binary import patch_binary
from .common import DUPLICATE_MODULE, write
from .multiclass import patch_multiclass_multi, patch_multiclass_single
from .ordinal import patch_ordinal_hetero, patch_ordinal_multi, patch_ordinal_single
from .tests_ci import add_tests_and_ci


def main() -> None:
    write("src/bochan/acquisition/_duplicate_exclusion.py", DUPLICATE_MODULE)
    patch_binary()
    patch_ordinal_single()
    patch_ordinal_multi()
    patch_ordinal_hetero(
        "src/bochan/acquisition/ordinal/active_learning/hetero_single_output.py",
        multi_output=False,
    )
    patch_ordinal_hetero(
        "src/bochan/acquisition/ordinal/active_learning/hetero_multi_output.py",
        multi_output=True,
    )
    patch_multiclass_single()
    patch_multiclass_multi()
    add_tests_and_ci()


if __name__ == "__main__":
    main()
