"""Deterministic random-stream infrastructure for APHREH-ADSMap 2.0.0.

Implementation step 1.7; architecture v1.2, supplied 2026-09-24.

References:
* AS Section 7: random number generator family and deterministic sub-seed
  derivation are implementation conventions that must be frozen and versioned.
* AS 9.5: every surface cell receives a deterministic sub-seed derived from
  surface master seed, realization identity, data-view identity, q, lag, and
  derivation-scheme version; execution order and worker count must not alter the
  stream assigned to a cell.
* PO5.2-5.3: dependence-preserving null realizations require reproducible random
  rearrangements.
* CM3, CM4.6, CM6.2, CM16.2, CM25.1, CM26: calibration retains a master random
  seed and deterministic sub-seed scheme and freezes the random-seed
  configuration for the complete execution.

Frozen conventions
------------------
``PYTHON_RANDOM_MT19937_V1``
    Uses a fresh ``random.Random`` instance seeded with a non-negative integer
    through the standard-library version-2 seeding API. No module-global random
    state is used.

``APHREH_SHA256_SUBSEED_V1``
    Derives a 256-bit non-negative integer sub-seed by SHA-256 hashing a
    domain-separated canonical JSON payload. Canonical JSON is provided by
    ``utils.hashing.APHREH_CANONICAL_JSON_V1``. The complete SHA-256 digest is
    interpreted as an unsigned big-endian integer; no truncation is applied.

These identifiers are provenance values. A caller requesting a different
scheme is rejected rather than silently receiving the current algorithm under a
misleading version label.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from math import isfinite
from random import Random

from .hashing import CANONICAL_SERIALIZATION_ID, canonical_json_bytes

RNG_FAMILY_ID = "PYTHON_RANDOM_MT19937_V1"
SUBSEED_DERIVATION_ID = "APHREH_SHA256_SUBSEED_V1"
SUBSEED_BITS = 256

# Prefixing the canonical payload prevents the same JSON bytes from being
# confused with an ordinary object hash or another future hash-derived value.
_SUBSEED_DOMAIN_PREFIX = b"APHREH-ADSMap\x00subseed\x00v1\x00"

SURFACE_CELL_NAMESPACE = "SURFACE_CELL"
BOOTSTRAP_NAMESPACE = "BOOTSTRAP"
CALIBRATION_NAMESPACE = "CALIBRATION"
OPTIMIZATION_NULL_NAMESPACE = "OPTIMIZATION_NULL"


class RandomnessConfigurationError(ValueError):
    """Raised when a seed or deterministic stream identity is invalid."""


def _require_seed(value: int, name: str) -> int:
    """Return a validated non-negative integer seed, rejecting booleans."""
    if type(value) is not int or value < 0:
        raise RandomnessConfigurationError(
            f"{name}: expected a non-negative integer"
        )
    return value


def _require_text(value: str, name: str) -> str:
    """Return nonblank text used as an immutable stream identity component."""
    if type(value) is not str or not value.strip():
        raise RandomnessConfigurationError(f"{name}: expected nonblank text")
    return value


def _require_scheme(scheme_version: str) -> str:
    """Reject unknown derivation versions rather than silently aliasing them."""
    _require_text(scheme_version, "scheme_version")
    if scheme_version != SUBSEED_DERIVATION_ID:
        raise RandomnessConfigurationError(
            f"scheme_version: unsupported derivation scheme {scheme_version!r}"
        )
    return scheme_version


def rng_from_seed(seed: int) -> Random:
    """Return an isolated MT19937 RNG for one already resolved seed.

    A fresh instance is returned every time. Consequently, unrelated calls,
    execution order, worker count, and module-global random state cannot advance
    this stream.
    """
    resolved = _require_seed(seed, "seed")
    rng = Random()
    rng.seed(resolved, version=2)
    return rng


def derive_subseed(
        master_seed: int,
        *,
        namespace: str,
        identity: Mapping[str, object],
        scheme_version: str = SUBSEED_DERIVATION_ID,
) -> int:
    """Derive one deterministic 256-bit sub-seed from immutable identities.

    ``namespace`` separates otherwise identical identities used for different
    stochastic purposes, for example bootstrap versus calibration. ``identity``
    must be canonically serializable by :mod:`utils.hashing`; mapping key order
    therefore has no effect, while scientifically meaningful value changes do.
    """
    master = _require_seed(master_seed, "master_seed")
    stream_namespace = _require_text(namespace, "namespace")
    scheme = _require_scheme(scheme_version)
    if not isinstance(identity, Mapping):
        raise RandomnessConfigurationError("identity: expected a mapping")

    payload = {
        "canonical_serialization": CANONICAL_SERIALIZATION_ID,
        "identity": identity,
        "master_seed": master,
        "namespace": stream_namespace,
        "scheme_version": scheme,
    }
    digest = sha256(_SUBSEED_DOMAIN_PREFIX + canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def rng_for_identity(
        master_seed: int,
        *,
        namespace: str,
        identity: Mapping[str, object],
        scheme_version: str = SUBSEED_DERIVATION_ID,
) -> Random:
    """Return an isolated RNG for a deterministic master-seed/identity pair."""
    return rng_from_seed(
        derive_subseed(
            master_seed,
            namespace=namespace,
            identity=identity,
            scheme_version=scheme_version,
        )
    )


def derive_surface_cell_subseed(
        surface_master_seed: int,
        *,
        realization_id: str,
        data_view_identity: str,
        q: float,
        lag: int,
        derivation_scheme_version: str = SUBSEED_DERIVATION_ID,
) -> int:
    """Derive the AS 9.5 random seed assigned to one ``(q, lag)`` cell.

    This helper implements the exact surface-cell identity required by the
    architecture. It deliberately does not include worker number, task order,
    process identifier, or runtime output path.
    """
    _require_seed(surface_master_seed, "surface_master_seed")
    _require_text(realization_id, "realization_id")
    _require_text(data_view_identity, "data_view_identity")
    _require_scheme(derivation_scheme_version)

    if type(q) not in (int, float):
        raise RandomnessConfigurationError("q: expected a finite percentile")
    q_float = float(q)
    if not isfinite(q_float) or not 0.0 <= q_float <= 100.0:
        raise RandomnessConfigurationError("q: expected a finite percentile in [0, 100]")
    if type(lag) is not int or lag < 0:
        raise RandomnessConfigurationError("lag: expected a non-negative integer")

    return derive_subseed(
        surface_master_seed,
        namespace=SURFACE_CELL_NAMESPACE,
        identity={
            "data_view_identity": data_view_identity,
            "lag": lag,
            "q": q_float,
            "realization_id": realization_id,
        },
        scheme_version=derivation_scheme_version,
    )


def surface_cell_rng(
        surface_master_seed: int,
        *,
        realization_id: str,
        data_view_identity: str,
        q: float,
        lag: int,
        derivation_scheme_version: str = SUBSEED_DERIVATION_ID,
) -> Random:
    """Return the isolated RNG assigned to one surface cell under AS 9.5."""
    return rng_from_seed(
        derive_surface_cell_subseed(
            surface_master_seed,
            realization_id=realization_id,
            data_view_identity=data_view_identity,
            q=q,
            lag=lag,
            derivation_scheme_version=derivation_scheme_version,
        )
    )


__all__ = [
    "RNG_FAMILY_ID",
    "SUBSEED_DERIVATION_ID",
    "SUBSEED_BITS",
    "SURFACE_CELL_NAMESPACE",
    "BOOTSTRAP_NAMESPACE",
    "CALIBRATION_NAMESPACE",
    "OPTIMIZATION_NULL_NAMESPACE",
    "RandomnessConfigurationError",
    "rng_from_seed",
    "derive_subseed",
    "rng_for_identity",
    "derive_surface_cell_subseed",
    "surface_cell_rng",
]
