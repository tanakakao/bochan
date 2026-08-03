"""Chemical formula parsing and formatting utilities."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

_ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(_ELEMENT_SYMBOLS)}

# Standard atomic weights (or bracketed-isotope mass approximations for unstable elements).
ATOMIC_WEIGHTS = dict(
    zip(
        _ELEMENT_SYMBOLS,
        (
            1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999, 18.998403163, 20.1797,
            22.98976928, 24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078,
            44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38,
            69.723, 72.630, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224,
            92.90637, 95.95, 98.0, 101.07, 102.90550, 106.42, 107.8682, 112.414, 114.818, 118.710,
            121.760, 127.60, 126.90447, 131.293, 132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242,
            145.0, 150.36, 151.964, 157.25, 158.92535, 162.500, 164.93033, 167.259, 168.93422, 173.045,
            174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592,
            204.38, 207.2, 208.98040, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0377,
            231.03588, 238.02891, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,
            258.0, 259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0,
            282.0, 285.0, 286.0, 289.0, 290.0, 293.0, 294.0, 294.0,
        ),
        strict=True,
    )
)

_TOKEN_RE = re.compile(r"[A-Z][a-z]?|(?:\d+(?:\.\d*)?|\.\d+)|[()\[\]]")
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")


def _tokenize(formula: str) -> list[str]:
    compact = re.sub(r"\s+", "", formula)
    if not compact:
        raise ValueError("formula must not be empty.")
    tokens = _TOKEN_RE.findall(compact)
    if "".join(tokens) != compact:
        raise ValueError(f"Unsupported token in chemical formula {formula!r}.")
    return tokens


def _parse_number(tokens: Sequence[str], index: int) -> tuple[float, int]:
    if index < len(tokens) and _NUMBER_RE.fullmatch(tokens[index]):
        value = float(tokens[index])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Stoichiometric coefficients must be finite and positive.")
        return value, index + 1
    return 1.0, index


def _parse_tokens(tokens: Sequence[str], index: int = 0, closing: str | None = None) -> tuple[dict[str, float], int]:
    composition: defaultdict[str, float] = defaultdict(float)
    closing_map = {"(": ")", "[": "]"}

    while index < len(tokens):
        token = tokens[index]
        if closing is not None and token == closing:
            return dict(composition), index + 1
        if token in {")", "]"}:
            raise ValueError(f"Unexpected closing bracket {token!r}.")
        if token in closing_map:
            nested, index = _parse_tokens(tokens, index + 1, closing_map[token])
            multiplier, index = _parse_number(tokens, index)
            for element, amount in nested.items():
                composition[element] += amount * multiplier
            continue
        if _NUMBER_RE.fullmatch(token):
            raise ValueError("A coefficient is only allowed after an element/group or before a hydrate segment.")
        if token not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element symbol {token!r}.")
        amount, index = _parse_number(tokens, index + 1)
        composition[token] += amount

    if closing is not None:
        raise ValueError(f"Missing closing bracket {closing!r}.")
    return dict(composition), index


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a chemical formula into element amounts.

    Supports nested ``()`` / ``[]`` groups, decimal stoichiometry, and hydrate
    segments separated by ``·``. Site labels are not inferred from a formula.
    """

    total: defaultdict[str, float] = defaultdict(float)
    for raw_segment in formula.split("·"):
        segment = raw_segment.strip()
        if not segment:
            raise ValueError(f"Invalid hydrate separator placement in {formula!r}.")
        tokens = _tokenize(segment)
        multiplier = 1.0
        if tokens and _NUMBER_RE.fullmatch(tokens[0]):
            multiplier = float(tokens[0])
            tokens = tokens[1:]
            if not tokens:
                raise ValueError(f"Missing formula after hydrate multiplier in {formula!r}.")
        parsed, end = _parse_tokens(tokens)
        if end != len(tokens):
            raise ValueError(f"Could not fully parse formula {formula!r}.")
        for element, amount in parsed.items():
            total[element] += multiplier * amount
    return dict(total)


def normalize_composition(
    composition: Mapping[str, float],
    *,
    mode: str = "atomic_fraction",
    atomic_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Normalize element amounts as raw, atomic-fraction, or weight-fraction values."""

    clean = {str(element): float(amount) for element, amount in composition.items() if float(amount) > 0.0}
    if not clean:
        raise ValueError("composition must contain at least one positive amount.")
    if any(not math.isfinite(value) for value in clean.values()):
        raise ValueError("composition amounts must be finite.")

    normalized_mode = mode.lower()
    if normalized_mode in {"none", "raw", "stoichiometric"}:
        return clean
    if normalized_mode in {"atomic_fraction", "atomic", "fraction"}:
        denominator = sum(clean.values())
        return {element: amount / denominator for element, amount in clean.items()}
    if normalized_mode in {"weight_fraction", "weight", "mass_fraction"}:
        weights = dict(ATOMIC_WEIGHTS if atomic_weights is None else atomic_weights)
        missing = sorted(element for element in clean if element not in weights)
        if missing:
            raise KeyError(f"Atomic weights are missing for elements: {missing!r}.")
        masses = {element: amount * float(weights[element]) for element, amount in clean.items()}
        denominator = sum(masses.values())
        return {element: mass / denominator for element, mass in masses.items()}
    raise ValueError("mode must be one of 'none', 'atomic_fraction', or 'weight_fraction'.")


def element_order(elements: Sequence[str], *, hill: bool = False) -> list[str]:
    """Return a deterministic element order."""

    unique = list(dict.fromkeys(elements))
    if hill and "C" in unique:
        leading = [element for element in ("C", "H") if element in unique]
        remaining = sorted(element for element in unique if element not in {"C", "H"})
        return leading + remaining
    return sorted(unique, key=lambda element: (ATOMIC_NUMBERS.get(element, 10_000), element))


def format_formula(
    composition: Mapping[str, float],
    *,
    order: Sequence[str] | None = None,
    precision: int = 6,
    omit_one: bool = True,
    zero_tolerance: float = 1e-12,
) -> str:
    """Format element amounts as a deterministic canonical formula string."""

    if precision < 0:
        raise ValueError("precision must be non-negative.")
    clean = {str(element): float(value) for element, value in composition.items() if abs(float(value)) > zero_tolerance}
    if any(value < -zero_tolerance or not math.isfinite(value) for value in clean.values()):
        raise ValueError("composition values must be finite and non-negative.")
    if not clean:
        raise ValueError("composition must contain at least one non-zero value.")

    ordered = list(order) if order is not None else element_order(list(clean))
    ordered += [element for element in element_order(list(clean)) if element not in ordered]
    parts: list[str] = []
    for element in ordered:
        value = clean.get(element, 0.0)
        if value <= zero_tolerance:
            continue
        rounded = round(value, precision)
        if omit_one and math.isclose(rounded, 1.0, rel_tol=0.0, abs_tol=10 ** (-precision)):
            suffix = ""
        else:
            suffix = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
        parts.append(f"{element}{suffix}")
    return "".join(parts)
