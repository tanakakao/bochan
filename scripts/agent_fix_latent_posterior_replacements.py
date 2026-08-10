from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt"}

REPLACEMENTS = {
    "_get_binary_mc_latent_posterioror_probability_samples": "_get_binary_mc_posterior_for_probability_samples",
    "_latent_posterioror_submodel": "_posterior_for_submodel",
    "latent_posterioramily": "posterior_family",
    '("latent_posterior", "latent_posterior", "latent_posterior")': '("latent_posterior",)',
    '("latent_posterior", "latent_posterior")': '("latent_posterior",)',
    "latent_posterior / latent_posterior / latent_posterior": "latent_posterior",
    "latent_posterior / latent_posterior": "latent_posterior",
    "latent_posterior(X), or latent_posterior(X).": "latent_posterior(X).",
    "latent_posterior / latent_posterior / latent_posterior /": "latent_posterior /",
    "修正済み latent_posterior / latent_posterior / latent_posterior": "canonical latent_posterior",
    "  - model.latent_posterior(X)\n  - model.latent_posterior(X)\n  - model.latent_posterior(X)\n": "  - model.latent_posterior(X)\n",
}

FORBIDDEN = (
    "latent_posterioror",
    "latent_posterioramily",
    '("latent_posterior", "latent_posterior"',
    "latent_posterior / latent_posterior",
    "latent_posterior(X), or latent_posterior(X)",
)


def iter_text_files():
    for root_name in ("src", "tests", "docs"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path


def main() -> None:
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        new = text
        for old, replacement in REPLACEMENTS.items():
            new = new.replace(old, replacement)
        if new != text:
            path.write_text(new, encoding="utf-8")

    offenders: list[str] = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    if offenders:
        raise RuntimeError("mechanical latent-posterior replacements remain:\n" + "\n".join(offenders))


if __name__ == "__main__":
    main()
