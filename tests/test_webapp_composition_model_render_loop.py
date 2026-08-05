from pathlib import Path


SOURCE = Path("web/src/compositionExtension.ts")


def test_composition_panels_use_render_signatures_instead_of_serialized_inner_html() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "const panelRenderSignatures = new WeakMap<HTMLElement, string>();" in source
    assert "panelRenderSignatures.get(host) === html" in source
    assert "panelRenderSignatures.set(replacement, html)" in source
    assert "host.innerHTML === html" not in source


def test_composition_observer_does_not_observe_its_own_panel_replacement() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "let compositionObserver: MutationObserver | null = null;" in source
    assert "compositionObserver?.disconnect();" in source
    assert "finally {\n    observeCompositionMutations();\n  }" in source
    assert "records.some(mutationAffectsComposition)" in source
    assert "COMPOSITION_ANCHOR_SELECTOR" in source
