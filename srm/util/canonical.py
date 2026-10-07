"""Canonical JSON (RFC 8785 JCS), hashing and deterministic seeds (09 §2, §4.1; D-005)."""

from __future__ import annotations

import base64
import enum
import hashlib
import json
import math
from typing import Any

import numpy as np


def _format_number(value: float) -> str:
    """Serialize a float the way ECMAScript ``Number.prototype.toString`` does."""
    if not math.isfinite(value):
        raise ValueError("canonical JSON does not allow NaN or Infinity")
    if value == 0.0:
        return "0"
    negative = value < 0
    mag = -value if negative else value
    text = repr(mag)
    if "e" in text:
        mantissa, exp_text = text.split("e")
        exponent = int(exp_text)
    else:
        mantissa, exponent = text, 0
    int_part, _, frac_part = mantissa.partition(".")
    all_digits = int_part + frac_part
    lead = len(all_digits) - len(all_digits.lstrip("0"))
    digits = all_digits.lstrip("0").rstrip("0") or "0"
    n = len(int_part) + exponent - lead
    k = len(digits)
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        sign = "+" if e >= 0 else "-"
        out = (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + sign + str(abs(e))
    return "-" + out if negative else out


def _serialize(obj: Any, out: list[str]) -> None:
    if obj is None:
        out.append("null")
    elif obj is True:
        out.append("true")
    elif obj is False:
        out.append("false")
    elif isinstance(obj, enum.Enum):
        _serialize(obj.value, out)
    elif isinstance(obj, (int, np.integer)):
        out.append(str(int(obj)))
    elif isinstance(obj, (float, np.floating)):
        f = float(obj)
        if f.is_integer() and abs(f) < 1e21:
            out.append(str(int(f)))
        else:
            out.append(_format_number(f))
    elif isinstance(obj, str):
        out.append(json.dumps(obj, ensure_ascii=False))
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for i, item in enumerate(obj):
            if i:
                out.append(",")
            _serialize(item, out)
        out.append("]")
    elif isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON object keys must be str, got {type(key).__name__}")
        out.append("{")
        for i, key in enumerate(sorted(obj, key=lambda s: s.encode("utf-16-be"))):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _serialize(obj[key], out)
        out.append("}")
    else:
        raise TypeError(f"type {type(obj).__name__} is not canonical-JSON serializable")


def canonical_json(obj: Any) -> str:
    """Return the RFC 8785 canonical JSON text of ``obj``."""
    out: list[str] = []
    _serialize(obj, out)
    return "".join(out)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).digest()


def canonical_hash(obj: Any) -> str:
    """SHA-256 (hex) of the canonical JSON of ``obj``."""
    return sha256_hex(canonical_json(obj))


def b32_hash(obj: Any, length: int = 26) -> str:
    """Lowercase, unpadded base32 of SHA-256(canonical JSON), truncated (09 §2.1)."""
    raw = sha256_bytes(canonical_json(obj))
    return base64.b32encode(raw).decode("ascii").lower().rstrip("=")[:length]


def derive_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary JSON-serializable parts."""
    return int.from_bytes(sha256_bytes(canonical_json(list(parts)))[:8], "big") >> 1


def array_digest(*arrays: np.ndarray) -> str:
    """SHA-256 over dtype, shape and bytes of the given arrays (weight digests)."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()
