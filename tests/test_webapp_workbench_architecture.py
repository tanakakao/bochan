from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web/src/App.tsx"
CONTEXT = ROOT / "web/src/context/WorkbenchContext.tsx"


def test_app_only_composes_workbench_shell_components() -> None:
    app = APP.read_text(encoding="utf-8")
    shell = (ROOT / "web/src/components/workbench/useWorkbenchShell.ts").read_text(
        encoding="utf-8"
    )
    page_registry = (ROOT / "web/src/components/workbench/workbenchPages.ts").read_text(
        encoding="utf-8"
    )

    for component in (
        "WorkbenchHeader",
        "WorkbenchLeftRail",
        "WorkbenchContextRail",
        "WorkbenchStatusBar",
        "WorkbenchBusyOverlay",
        "WorkbenchErrorAlert",
    ):
        assert component in app

    assert "useWorkbenchShell" in app
    assert "useEffect" not in app
    assert "useState" not in app
    assert "window.location.hash" not in app
    assert "window.localStorage" not in app
    assert "ConversationPage" not in app
    assert "ExperimentPage" not in app

    assert "window.location.hash" in shell
    assert "CONTEXT_COLLAPSE_KEY" in shell
    assert 'STEPS.filter(([id]) => id !== "logs")' in shell
    assert 'import ConversationPage from "../../pages/ConversationPage"' in page_registry
    assert 'import ExperimentPage from "../../pages/ExperimentPage"' in page_registry


def test_workbench_context_composes_domain_state_hooks() -> None:
    context = CONTEXT.read_text(encoding="utf-8")
    runtime = (ROOT / "web/src/context/useWorkbenchRuntimeState.ts").read_text(
        encoding="utf-8"
    )
    selection = (ROOT / "web/src/context/useWorkbenchSelectionState.ts").read_text(
        encoding="utf-8"
    )
    settings = (ROOT / "web/src/context/useWorkbenchRunSettings.ts").read_text(
        encoding="utf-8"
    )
    validation = (ROOT / "web/src/context/workbenchValidation.ts").read_text(
        encoding="utf-8"
    )
    defaults = (ROOT / "web/src/context/workbenchDefaults.ts").read_text(
        encoding="utf-8"
    )
    types = (ROOT / "web/src/context/workbenchTypes.ts").read_text(encoding="utf-8")

    for hook in (
        "useWorkbenchRuntimeState",
        "useWorkbenchSelectionState",
        "useWorkbenchRunSettings",
        "useWorkbenchResultState",
    ):
        assert hook in context

    assert "deriveWorkbenchState" in context
    assert "useState(" not in context
    assert "fetchHealth" not in context
    assert "getColumnClassValues" not in context
    assert "MODEL_OPTIONS" not in context

    assert "fetchHealth" in runtime
    assert "bochan-theme" in runtime
    assert "toggleFeature" in selection
    assert "toggleTarget" in selection
    assert "const [inputPerturbation, setInputPerturbation] = useState(false);" in settings
    assert "validateTargetCandidateSetting" in validation
    assert "MODEL_OPTIONS" in validation
    assert "createInitialSelectionState" in defaults
    assert "export interface WorkbenchContextValue" in types
    assert "export const STEPS" in types
