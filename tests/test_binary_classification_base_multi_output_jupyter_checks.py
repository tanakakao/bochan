from __future__ import annotations

"""Jupyter helpers for binary classification base multi-output tests.

The pytest smoke tests in ``test_binary_classification_base_multi_output`` are
kept compact.  This module provides notebook-oriented runners that mirror
``test_binary_classification_base_single_output.run_jupyter_all_checks``:

1. forward checks for normal and mixed multi-output models,
2. optimize_acqf / optimize_acqf_mixed checks for all acquisition cases, and
3. optimizer × constraint compatibility checks for representative acquisition
   cases, including base / torch / evo optimizers and step / linear / k-sparse
   repair cases.
"""

from typing import Any

import torch
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from tests.test_binary_classification_base_multi_output import (
    N_OUTPUTS,
    _optimizer_constraint_scenarios,
    create_multi_output_binary_model_bundle,
    multi_output_acquisition_cases,
    run_jupyter_forward_check,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
    assert_candidates_in_bounds,
    assert_optimizer_compatibility_result,
    maybe_suppress_botorch_initial_warnings,
    optimize_mixed_with_case,
    optimize_with_case,
    print_linear_constraint_diagnostics,
)


def _print_failure_summary(failed_cases: list[tuple[str, Exception]]) -> None:
    print("=" * 100)
    if failed_cases:
        print(f"failed_cases={len(failed_cases)}")
        for case_id, exc in failed_cases:
            print(f"  - {case_id}: {type(exc).__name__}: {exc}")
    else:
        print("all checks passed.")
    print("=" * 100)


def run_jupyter_optimize_acqf_all_acquisitions_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
    verbose_ok_detail: bool = False,
) -> dict[str, Any]:
    """通常 multi-output model で全 acquisition を optimize_acqf に通す。"""
    bundle = create_multi_output_binary_model_bundle(
        cat=False,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]

    cases = multi_output_acquisition_cases(model, train_x)
    print("=" * 100)
    print("Jupyter multi-output optimize_acqf check: all acquisitions")
    print(
        f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, "
        f"num_acquisitions={len(cases)}"
    )
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for acq_cls, kwargs, case_id in cases:
        display_id = f"optimize_acqf__{case_id}"
        try:
            acqf = acq_cls(model=model, **kwargs)
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
                cands, acq_value = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=q,
                    sequential=True,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )

            assert cands.shape == torch.Size([q, train_x.shape[-1]]), (
                f"{display_id}: 候補点 shape が想定外です。"
                f"expected={torch.Size([q, train_x.shape[-1]])}, actual={cands.shape}"
            )
            assert torch.isfinite(cands).all(), f"{display_id}: 候補点に NaN/inf が含まれます。cands={cands}"
            assert torch.isfinite(acq_value).all(), f"{display_id}: acq_value に NaN/inf が含まれます。acq_value={acq_value}"
            assert_candidates_in_bounds(cands=cands, bounds=bounds)

            if verbose_ok_detail:
                print(f"[OK] {display_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}")
            else:
                print(f"[OK] {display_id}")

        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
    verbose_ok_detail: bool = False,
) -> dict[str, Any]:
    """mixed multi-output model で全 acquisition を optimize_acqf_mixed に通す。"""
    bundle = create_multi_output_binary_model_bundle(
        cat=True,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    assert cat_dims == [d]
    cat_id = cat_dims[0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)

    cases = multi_output_acquisition_cases(model, train_x)
    print("=" * 100)
    print("Jupyter mixed multi-output optimize_acqf_mixed check: all acquisitions")
    print(
        f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, "
        f"cat_dims={cat_dims}, num_acquisitions={len(cases)}"
    )
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for acq_cls, kwargs, case_id in cases:
        display_id = f"optimize_acqf_mixed__{case_id}"
        try:
            acqf = acq_cls(model=model, **kwargs)
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
                cands, acq_value = optimize_acqf_mixed(
                    acq_function=acqf,
                    bounds=bounds,
                    q=q,
                    fixed_features_list=fixed_features_list,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )

            assert cands.shape == torch.Size([q, train_x.shape[-1]]), (
                f"{display_id}: 候補点 shape が想定外です。"
                f"expected={torch.Size([q, train_x.shape[-1]])}, actual={cands.shape}"
            )
            assert torch.isfinite(cands).all(), f"{display_id}: 候補点に NaN/inf が含まれます。cands={cands}"
            assert torch.isfinite(acq_value).all(), f"{display_id}: acq_value に NaN/inf が含まれます。acq_value={acq_value}"
            assert_candidates_in_bounds(cands=cands, bounds=bounds)
            assert torch.isin(cands[:, cat_id], cat_values).all(), (
                f"{display_id}: カテゴリ候補が想定値以外です。"
                f"expected={cat_values}, actual={cands[:, cat_id]}"
            )

            if verbose_ok_detail:
                print(
                    f"[OK] {display_id} cands.shape={tuple(cands.shape)} "
                    f"cat={cands[:, cat_id].detach().cpu().tolist()} acq_value={acq_value}"
                )
            else:
                print(f"[OK] {display_id}")

        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_optimizer_constraint_compatibility_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    q: int = 2,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    """通常 multi-output model の optimizer × constraint compatibility を確認する。"""
    if d < 5:
        raise ValueError("constraint compatibility check では d >= 5 が必要です。")

    bundle = create_multi_output_binary_model_bundle(
        cat=False,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    scenarios = _optimizer_constraint_scenarios(
        model=model,
        train_x=train_x,
        bounds=bounds,
        mixed=False,
        full_matrix=full_matrix,
    )

    print("=" * 100)
    print("Jupyter multi-output optimizer / constraint compatibility check")
    print(
        f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, "
        f"full_matrix={full_matrix}, num_cases={len(scenarios)}"
    )
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in scenarios:
        try:
            acqf = acq_cls(model=model, **kwargs)
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
                cands, acq_value = optimize_with_case(
                    acqf=acqf,
                    bounds=bounds,
                    q=q,
                    optimize_func=optimize_func,
                    optimize_method=optimize_method,
                    constraint_case=constraint_case,
                    num_restarts=2,
                    raw_samples=16,
                    maxiter=10,
                )

            assert_optimizer_compatibility_result(
                cands=cands,
                acq_value=acq_value,
                bounds=bounds,
                q=q,
                d=train_x.shape[-1],
                constraint_case=constraint_case,
                case_id=case_id,
            )

            if verbose_ok_detail:
                print(f"[OK] {case_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}")
            else:
                print(f"[OK] {case_id}")

            if verbose_candidates:
                print(f"     cands={cands}")
                if train_x.shape[-1] >= 5:
                    print(f"     sum_0_1_2={cands[:, :3].sum(dim=1)}")
                    print(f"     sum_3_4={cands[:, 3:5].sum(dim=1)}")

            if constraint_case["case_id"] != "none":
                print_linear_constraint_diagnostics(
                    cands=cands,
                    equality_constraints=constraint_case["equality_constraints"],
                    inequality_constraints=constraint_case["inequality_constraints"],
                    inequality_sense=constraint_case.get("inequality_sense", "le"),
                    show_all=verbose_constraints,
                )

        except Exception as exc:
            print(f"[NG] {case_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((case_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_mixed_optimizer_constraint_compatibility_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    q: int = 2,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    """mixed multi-output model の optimizer × constraint compatibility を確認する。"""
    if d < 5:
        raise ValueError("constraint compatibility check では d >= 5 が必要です。")

    bundle = create_multi_output_binary_model_bundle(
        cat=True,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    assert cat_dims == [d]
    cat_id = cat_dims[0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)

    scenarios = _optimizer_constraint_scenarios(
        model=model,
        train_x=train_x,
        bounds=bounds,
        mixed=True,
        full_matrix=full_matrix,
    )

    print("=" * 100)
    print("Jupyter mixed multi-output optimizer / constraint compatibility check")
    print(
        f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, "
        f"cat_dims={cat_dims}, full_matrix={full_matrix}, num_cases={len(scenarios)}"
    )
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in scenarios:
        try:
            acqf = acq_cls(model=model, **kwargs)
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
                cands, acq_value = optimize_mixed_with_case(
                    acqf=acqf,
                    bounds=bounds,
                    q=q,
                    fixed_features_list=fixed_features_list,
                    optimize_func=optimize_func,
                    optimize_method=optimize_method,
                    constraint_case=constraint_case,
                    num_restarts=2,
                    raw_samples=16,
                    maxiter=10,
                )

            assert_optimizer_compatibility_result(
                cands=cands,
                acq_value=acq_value,
                bounds=bounds,
                q=q,
                d=train_x.shape[-1],
                constraint_case=constraint_case,
                case_id=case_id,
            )
            assert torch.isin(cands[:, cat_id], cat_values).all(), (
                f"{case_id}: カテゴリ候補が想定値以外です。"
                f"expected={cat_values}, actual={cands[:, cat_id]}"
            )

            if verbose_ok_detail:
                print(
                    f"[OK] {case_id} cands.shape={tuple(cands.shape)} "
                    f"cat={cands[:, cat_id].detach().cpu().tolist()} acq_value={acq_value}"
                )
            else:
                print(f"[OK] {case_id}")

            if verbose_candidates:
                print(f"     cands={cands}")
                if train_x.shape[-1] >= 5:
                    print(f"     sum_0_1_2={cands[:, :3].sum(dim=1)}")
                    print(f"     sum_3_4={cands[:, 3:5].sum(dim=1)}")
                print(f"     cat_values={cands[:, cat_id]}")

            if constraint_case["case_id"] != "none":
                print_linear_constraint_diagnostics(
                    cands=cands,
                    equality_constraints=constraint_case["equality_constraints"],
                    inequality_constraints=constraint_case["inequality_constraints"],
                    inequality_sense=constraint_case.get("inequality_sense", "le"),
                    show_all=verbose_constraints,
                )

        except Exception as exc:
            print(f"[NG] {case_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((case_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_all_checks(
    *,
    num_epochs: int = 4,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    q: int = 2,
    run_optimize: bool = True,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_forward_detail: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> None:
    """multi-output binary classification の Jupyter 一括確認 helper。

    Args:
        num_epochs: 各 single-output submodel の fit epoch 数。
        n: 初期データ数。
        d: 連続変数の入力次元。mixed では全体次元は ``d + 1``。
        m: 出力数。
        q: q-batch size。
        run_optimize:
            False の場合は forward check のみ。
            True の場合は以下も実行する。

            1. 通常 multi-output の全 acquisition に対する ``optimize_acqf``
            2. mixed multi-output の全 acquisition に対する ``optimize_acqf_mixed``
            3. 通常 multi-output の representative acquisition × optimizer × constraint
            4. mixed multi-output の representative acquisition × optimizer × constraint

        full_matrix:
            optimizer / constraint compatibility で、代表 acquisition × optimizer × constraint
            の全組み合わせを回すかどうか。
        continue_on_error:
            True の場合、失敗しても次の case に進む。
        verbose_forward_detail:
            forward check の OK case で shape / min / max を表示する。
        verbose_ok_detail:
            optimize / constraint check の OK case で候補点 shape / acq_value を表示する。
        verbose_candidates:
            optimize / constraint check の OK case で候補点も表示する。
        verbose_constraints:
            制約診断で OK 制約も表示する。
        suppress_botorch_warnings:
            BoTorch の初期候補 warning を抑制する。
    """
    run_jupyter_forward_check(
        cat=False,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
        q=q,
        verbose_forward_detail=verbose_forward_detail,
    )
    run_jupyter_forward_check(
        cat=True,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
        q=q,
        verbose_forward_detail=verbose_forward_detail,
    )

    if run_optimize:
        run_jupyter_optimize_acqf_all_acquisitions_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
            verbose_ok_detail=verbose_ok_detail,
        )
        run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
            verbose_ok_detail=verbose_ok_detail,
        )
        run_jupyter_optimizer_constraint_compatibility_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )
        run_jupyter_mixed_optimizer_constraint_compatibility_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )

    print("all multi-output binary Jupyter checks passed.")
