from pathlib import Path

path = Path("src/bochan/models/components/layers/kernel_layers.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from linear_operator.operators import MaskedLinearOperator, MatmulLinearOperator\n",
    "from linear_operator.operators import MaskedLinearOperator\n",
    1,
)
old = '''        correction_rhs = train_covar.solve(
            test_train_observed.transpose(-1, -2)
        )
        return to_linear_operator(test_test_covar) + MatmulLinearOperator(
            test_train_observed,
            correction_rhs.mul(-1),
        )
'''
new = '''        test_train_dense = test_train_observed.to_dense()
        correction_rhs = train_covar.solve(test_train_dense.transpose(-1, -2))
        correction = test_train_dense @ correction_rhs
        return to_linear_operator(test_test_covar) + to_linear_operator(
            correction.mul(-1)
        )
'''
if old not in text:
    raise SystemExit("masked covariance solve block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
