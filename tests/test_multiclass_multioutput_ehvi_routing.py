from bochan.api.registry.acquisition import resolve_acqf_cls


def test_multiclass_multi_output_ehvi_uses_dedicated_acquisition() -> None:
    acqf_cls = resolve_acqf_cls(
        "ehvi",
        task_type="multiclass",
        model_type="base",
        multi_output=True,
    )
    assert "MultiOutputMulticlass" in acqf_cls.__name__
