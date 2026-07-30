"""
Version-compatibility guard between Gymnasium and stable-baselines3.

Gymnasium and stable-baselines3 (SB3) evolve on independent release schedules,
and SB3 typically lags behind Gymnasium's newest version. When a user upgrades
Gymnasium past what their installed SB3 supports, the breakage shows up as
cryptic errors deep inside SB3. This module reads SB3's *declared* Gymnasium
version requirement (its ``Requires-Dist`` metadata) and warns up front if the
installed Gymnasium falls outside that range, so the mismatch fails fast with a
clear, actionable message.

The check is best-effort and never raises during normal import: if SB3 is not
installed, or ``packaging`` is unavailable, or the metadata can't be parsed, the
check silently passes.
"""
from __future__ import annotations

import warnings


def _get_sb3_gymnasium_spec():
    """Return SB3's gymnasium specifier string (e.g. '<2.0,>=0.29.1') or None."""
    try:
        from importlib import metadata
    except Exception:  # pragma: no cover
        return None
    try:
        requires = metadata.distribution("stable_baselines3").requires or []
    except Exception:
        return None  # SB3 not installed -> nothing to check
    for req in requires:
        name = req.split(";")[0].strip()  # drop environment markers
        if name.lower().startswith("gymnasium"):
            spec = name[len("gymnasium"):].strip()
            return spec or None
    return None


def _get_gymnasium_version():
    try:
        from importlib import metadata
        return metadata.version("gymnasium")
    except Exception:
        return None


def verify_compatibility(raise_on_mismatch: bool = False) -> bool:
    """Warn (or raise) if installed Gymnasium is outside SB3's supported range.

    Returns True when compatible, or when the check cannot be performed (SB3 not
    installed, packaging unavailable, metadata unparseable). This means it is safe
    to call from package import: it only ever emits a warning unless explicitly
    asked to raise.
    """
    gym_version = _get_gymnasium_version()
    spec = _get_sb3_gymnasium_spec()
    if gym_version is None or not spec:
        return True

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except Exception:
        return True  # packaging unavailable -> can't verify, don't block

    specifier = SpecifierSet(spec)
    if specifier.contains(Version(gym_version), prereleases=True):
        return True

    msg = (
        f"slimevolleygym: installed gymnasium {gym_version} is outside "
        f"stable-baselines3's supported range ({spec}). SB3 (used by the training/"
        f"eval scripts) may break. Pin a compatible version, e.g. "
        f"pip install \"gymnasium{spec}\"."
    )
    if raise_on_mismatch:
        raise ImportError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return False
