"""Deterministic serialization and SHA-256 hashing for APHREH-ADSMap 2.0.0.

Implementation step 1.6; architecture v1.2, supplied 2026-09-24.

This module freezes the first concrete deterministic serialization convention
required by AS Section 7. It is infrastructure only: no estimator or
methodological calculation is implemented here.

Frozen convention ``APHREH_CANONICAL_JSON_V1``:
* JSON text is UTF-8 encoded without a byte-order mark.
* Mapping keys must be strings and are sorted lexicographically.
* Arrays preserve their input order; tuples and lists serialize identically.
* Whitespace outside JSON string values is omitted.
* Unicode characters are emitted directly (``ensure_ascii=False``).
* ``StrEnum``/``Enum`` values serialize through their documented wire value.
* Timezone-aware datetimes and dates serialize as ISO 8601 strings.
* Objects exposing ``to_dict()`` are serialized from that detached mapping.
* Non-finite floating-point values are rejected, never converted to null or a
  string sentinel. Negative zero is normalized to ``0.0`` so numerically equal
  zero values cannot acquire different hashes solely from their sign bit.

Source-file hashes are computed over the exact raw file bytes and therefore do
not apply JSON, newline, encoding, or Unicode normalization.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from os import PathLike
from pathlib import Path
from typing import Protocol, runtime_checkable


HASH_ALGORITHM = "SHA-256"
CANONICAL_SERIALIZATION_ID = "APHREH_CANONICAL_JSON_V1"
NONFINITE_SERIALIZATION_ID = "REJECT_NONFINITE_V1"
DEFAULT_FILE_CHUNK_SIZE = 1024 * 1024


class CanonicalSerializationError(ValueError):
    """Raised when a value cannot enter the frozen canonical JSON format."""


class DigestValidationError(ValueError):
    """Raised when an expected SHA-256 digest is structurally invalid."""


@runtime_checkable
class SupportsToDict(Protocol):
    """Structural protocol used by immutable domain and contract objects."""

    def to_dict(self) -> Mapping[str, object]:
        """Return a detached JSON-compatible representation."""


def _canonical_value(value: object, path: str = "$" ) -> object:
    """Convert one supported value tree to deterministic JSON-compatible data."""
    if value is None or type(value) in (str, bool, int):
        return value

    if type(value) is float:
        if not isfinite(value):
            raise CanonicalSerializationError(
                f"{path}: non-finite floats are forbidden by {NONFINITE_SERIALIZATION_ID}"
            )
        return 0.0 if value == 0.0 else value

    if isinstance(value, Enum):
        return _canonical_value(value.value, path)

    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise CanonicalSerializationError(
                f"{path}: datetime must be timezone-aware"
            )
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, SupportsToDict):
        exported = value.to_dict()
        if exported is value:
            raise CanonicalSerializationError(
                f"{path}: to_dict() must return a detached representation"
            )
        return _canonical_value(exported, path)

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalSerializationError(
                    f"{path}: mapping keys must be strings"
                )
            result[key] = _canonical_value(item, f"{path}.{key}")
        return result

    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise CanonicalSerializationError(
        f"{path}: unsupported canonical type {type(value).__name__}"
    )


def canonical_json(value: object) -> str:
    """Return canonical JSON text under ``APHREH_CANONICAL_JSON_V1``.

    The output is suitable as stable hash input. It is intentionally separate
    from ``PersistentModel.to_dict()``, which is a detached persistence/export
    representation rather than a canonical byte encoding.
    """
    canonical = _canonical_value(value)
    try:
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # Defensive boundary around stdlib JSON.
        raise CanonicalSerializationError(
            "value could not be encoded under the canonical JSON convention"
        ) from exc


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical JSON UTF-8 byte representation used for hashing."""
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return a lowercase SHA-256 hexadecimal digest for an in-memory byte buffer."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    return sha256(data).hexdigest()


def hash_object(value: object) -> str:
    """Hash a supported object through the frozen canonical JSON representation."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def hash_file(
    path: str | PathLike[str],
    *,
    chunk_size: int = DEFAULT_FILE_CHUNK_SIZE,
) -> str:
    """Hash the exact bytes of one file using streaming SHA-256.

    Filesystem errors are intentionally not converted into scientific or model
    validation errors. The source-loading layer decides how missing or
    inaccessible files map to SourceAvailability and validation reports.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256_digest(value: str) -> str:
    """Validate and normalize a SHA-256 hexadecimal digest to lowercase."""
    if type(value) is not str:
        raise DigestValidationError("SHA-256 digest must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise DigestValidationError("invalid SHA-256 hexadecimal digest")
    return normalized


def verify_file_hash(
    path: str | PathLike[str],
    expected_digest: str,
    *,
    chunk_size: int = DEFAULT_FILE_CHUNK_SIZE,
) -> bool:
    """Return whether one file's exact-byte SHA-256 equals the expected digest."""
    expected = normalize_sha256_digest(expected_digest)
    return hash_file(path, chunk_size=chunk_size) == expected


__all__ = [
    "HASH_ALGORITHM",
    "CANONICAL_SERIALIZATION_ID",
    "NONFINITE_SERIALIZATION_ID",
    "DEFAULT_FILE_CHUNK_SIZE",
    "CanonicalSerializationError",
    "DigestValidationError",
    "canonical_json",
    "canonical_json_bytes",
    "sha256_bytes",
    "hash_object",
    "hash_file",
    "normalize_sha256_digest",
    "verify_file_hash",
]
