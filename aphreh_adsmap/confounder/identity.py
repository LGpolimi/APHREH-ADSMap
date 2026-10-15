"""Deterministic identities for downstream confounder execution (CA3)."""
from __future__ import annotations

from pathlib import Path
from ..hashing import hash_object
from ..storage import sha256_file
from .configuration import ConfounderAdjustmentSettings
from .contracts import ConfounderCandidateDefinition, ConfounderPreprocessingResult


CONFOUNDER_IDENTITY_SCHEMA = "APHREH_CONFOUNDER_ANALYSIS_IDENTITY_CA3_V1"
CONFOUNDER_PROFILE_IDENTITY_SCHEMA = "APHREH_CONFOUNDER_PROFILE_IDENTITY_CA3_V1"
_PROFILING_CODE_FILES = ("preprocessing.py", "execution.py", "profiling.py")


def confounder_code_identity() -> dict[str, str]:
    """Hash the dedicated confounder package only.

    The completed reference analysis is protected separately by
    ``reference_identity``. Candidate profiling uses the unchanged core engine,
    so the confounder package hash captures the downstream code added by CA1-CA3
    without redefining the upstream reference result.
    """
    package = Path(__file__).resolve().parent
    base = package.parent.parent
    return {
        str(path.relative_to(base)): sha256_file(path)
        for path in sorted(package.glob("*.py"))
    }


def confounder_profiling_code_identity() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    base = package.parent.parent
    result = {}
    for name in _PROFILING_CODE_FILES:
        path = package / name
        if path.is_file():
            result[str(path.relative_to(base))] = sha256_file(path)
    return result


def confounder_identity(
    reference_identity: str,
    settings: ConfounderAdjustmentSettings,
    candidate: ConfounderCandidateDefinition,
    preprocessing: ConfounderPreprocessingResult,
) -> str:
    payload = {
        "schema": CONFOUNDER_IDENTITY_SCHEMA,
        "reference_identity": reference_identity,
        "settings": settings.to_dict(),
        "candidate": candidate.to_dict(),
        "preprocessing": {
            "schema_version": preprocessing.schema_version,
            "spatial_representation": preprocessing.spatial_representation.value,
            "bsa_count": preprocessing.bsa_count,
            "date_count": preprocessing.date_count,
            "years": list(preprocessing.years),
            "source_hashes": dict(preprocessing.source_hashes),
        },
        "confounder_code": confounder_code_identity(),
    }
    return hash_object(payload)


def confounder_run_id(candidate_id: str, identity: str) -> str:
    return f"CONF_{candidate_id}_{identity[:20]}"


def profile_identity(
    *,
    reference_identity: str,
    candidate: ConfounderCandidateDefinition,
    preprocessing: ConfounderPreprocessingResult,
    held_out_year: int,
    training_years: tuple[int, ...],
    thresholds: tuple[float, ...],
    lags: tuple[int, ...],
) -> str:
    """Identity of one cross-fitted candidate-only surface.

    Caliper and later pairwise-adjustment settings are intentionally absent so
    CA3 profiling can be reused by CA4 sensitivity calculations when the
    upstream candidate data and reference estimator are unchanged.
    """
    payload = {
        "schema": CONFOUNDER_PROFILE_IDENTITY_SCHEMA,
        "reference_identity": reference_identity,
        "candidate": candidate.to_dict(),
        "preprocessing_source_hashes": dict(preprocessing.source_hashes),
        "held_out_year": int(held_out_year),
        "training_years": list(training_years),
        "thresholds": list(thresholds),
        "lags": list(lags),
        "profiling_code": confounder_profiling_code_identity(),
    }
    return hash_object(payload)


def profile_run_id(identity: str, held_out_year: int) -> str:
    return f"CONF_PROFILE_H{int(held_out_year)}_{identity[:20]}"
