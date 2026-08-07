from __future__ import annotations

from pathlib import Path


ENGINE_PATH = Path("src/bochan/api/engine.py")
API_INIT_PATH = Path("src/bochan/api/__init__.py")

ENGINE_OLD = '''    def _resolve_acquisition_config(self, acq_config: AcquisitionConfig) -> AcquisitionConfig:\n        if acq_config.acqf_cls is not None or acq_config.acqf_factory is not None:\n            return acq_config\n        self._check_fitted()\n        acqf_cls = resolve_acqf_cls(\n            acq_config.name,\n            self.acquisition_registry,\n            task_type=self.bundle.task_type,\n            model_type=self.bundle.model_type,\n            multi_output=bool(self.bundle.metadata.get("multi_output", False)),\n        )\n        return replace(acq_config, acqf_cls=acqf_cls)\n'''

ENGINE_NEW = '''    def _acquisition_routing_context(self) -> tuple[str, str, bool]:\n        """Resolve task/model/output shape used only for acquisition lookup.\n\n        A one-output ``HybridMultiOutputModel`` is still useful as a Web/API\n        compatibility wrapper, but it must not force acquisition lookup into\n        the multi-output family.  In that case the sole submodel defines the\n        task and model family while the acquisition is resolved as\n        single-output.\n        """\n        self._check_fitted()\n        bundle = self.bundle\n        task_type = str(bundle.task_type)\n        model_type = str(bundle.model_type)\n        multi_output = bool(bundle.metadata.get("multi_output", False))\n\n        if task_type != "hybrid":\n            return task_type, model_type, multi_output\n\n        sub_bundles = list(bundle.metadata.get("sub_bundles") or [])\n        if len(sub_bundles) == 1:\n            sub_bundle = sub_bundles[0]\n            return (\n                str(sub_bundle.task_type),\n                str(sub_bundle.model_type),\n                False,\n            )\n\n        specs = list(getattr(bundle.model, "specs", None) or [])\n        if len(specs) == 1:\n            return str(specs[0].task_type), model_type, False\n\n        return task_type, model_type, multi_output\n\n    def _resolve_acquisition_config(self, acq_config: AcquisitionConfig) -> AcquisitionConfig:\n        if acq_config.acqf_cls is not None or acq_config.acqf_factory is not None:\n            return acq_config\n        task_type, model_type, multi_output = self._acquisition_routing_context()\n        acqf_cls = resolve_acqf_cls(\n            acq_config.name,\n            self.acquisition_registry,\n            task_type=task_type,\n            model_type=model_type,\n            multi_output=multi_output,\n        )\n        return replace(acq_config, acqf_cls=acqf_cls)\n'''

API_INIT_OLD = '''    self._check_fitted()\n    acqf_cls = resolve_acqf_cls(\n        acq_config.name,\n        self.acquisition_registry,\n        task_type=self.bundle.task_type,\n        model_type=self.bundle.model_type,\n        multi_output=_infer_bundle_multi_output(self.bundle),\n    )\n    return replace(acq_config, acqf_cls=acqf_cls)\n'''

API_INIT_NEW = '''    self._check_fitted()\n    task_type, model_type, multi_output = self._acquisition_routing_context()\n    if task_type == str(self.bundle.task_type):\n        multi_output = _infer_bundle_multi_output(self.bundle)\n    acqf_cls = resolve_acqf_cls(\n        acq_config.name,\n        self.acquisition_registry,\n        task_type=task_type,\n        model_type=model_type,\n        multi_output=multi_output,\n    )\n    return replace(acq_config, acqf_cls=acqf_cls)\n'''


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(path, "already patched")
        return
    if old not in text:
        raise RuntimeError(f"Expected routing block was not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)


def main() -> None:
    _patch(ENGINE_PATH, ENGINE_OLD, ENGINE_NEW)
    _patch(API_INIT_PATH, API_INIT_OLD, API_INIT_NEW)


if __name__ == "__main__":
    main()
