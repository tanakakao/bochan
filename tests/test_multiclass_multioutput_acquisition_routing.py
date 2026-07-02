from bochan.api.acquisition_registry import resolve_acqf_cls


def test_multiclass_multi_output_bald_uses_dedicated_acquisition() -> None:
    acqf_cls = resolve_acqf_cls(
        "bald",
        task_type="multiclass",
        model_type="base",
        multi_output=True,
    )
    assert acqf_cls.__name__ == "qMultiOutputMulticlassBALD"
