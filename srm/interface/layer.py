"""The interface layer object: code space, value encoder, registries, versioning, ABI hash (03 §1)."""

from __future__ import annotations

from dataclasses import dataclass

from srm.config.build_config import BuildConfig
from srm.interface.codespace import CodeSpace
from srm.interface.messages import MESSAGE_SCHEMA_VERSION
from srm.interface.primitives import primitive_signature_digest
from srm.interface.registries import (
    ConceptTypeRegistry,
    PropertyRegistry,
    RelationRegistry,
    RoleRegistry,
    ValueTypeRegistry,
)
from srm.interface.values import ValueEncoder
from srm.util.canonical import canonical_hash, derive_seed


@dataclass
class InterfaceVersion:
    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    @staticmethod
    def parse(text: str) -> "InterfaceVersion":
        major, minor = text.split(".")
        return InterfaceVersion(int(major), int(minor))


class InterfaceLayer:
    """Everything components share; frozen after bootstrap stage S1 (03 §1.4)."""

    def __init__(self, config: BuildConfig) -> None:
        ic = config.interface
        self.config = config
        seed = derive_seed(config.meta.model_family_id, config.meta.seed, "interface")
        self.codespace = CodeSpace(ic.B, ic.L, ic.d, ic.d_k, ic.r, ic.n_b, seed)
        self.values = ValueEncoder(self.codespace, ic.scalar_buckets)
        self.roles = RoleRegistry(self.codespace)
        self.roles.populate(ic.N_role)
        self.properties = PropertyRegistry(self.codespace)
        self.relations = RelationRegistry(self.codespace)
        self.concept_types = ConceptTypeRegistry(self.codespace)
        self.concept_types.populate()
        self.value_types = ValueTypeRegistry(self.codespace)
        self.value_types.populate()
        self.version = InterfaceVersion.parse(ic.version)
        self.frozen = False

    @property
    def registries(self) -> dict[str, object]:
        return {
            "role": self.roles, "property": self.properties, "relation": self.relations,
            "concept_type": self.concept_types, "value_type": self.value_types,
        }

    def freeze(self) -> None:
        """Freeze projection/embedding tables and existing registry entries."""
        self.codespace.freeze()
        for reg in self.registries.values():
            reg.freeze()  # type: ignore[attr-defined]
        self.frozen = True

    def note_minor_addition(self) -> None:
        """Appending registry entries after the freeze bumps the MINOR version (03 §1.3)."""
        if self.frozen:
            self.version = InterfaceVersion(self.version.major, self.version.minor + 1)

    def abi_description(self) -> dict[str, object]:
        cs = self.codespace
        return {
            "B": cs.B, "L": cs.L, "d": cs.d, "d_k": cs.d_k,
            "band_layout": [list(b) for b in cs.band_layout],
            "codespace_digest": cs.digest(),
            "registry_baselines": {
                name: reg.digest(baseline_only=True)  # type: ignore[attr-defined]
                for name, reg in sorted(self.registries.items())
            },
            "message_schema_version": MESSAGE_SCHEMA_VERSION,
            "primitive_signatures": primitive_signature_digest(),
            "interface_major": self.version.major,
        }

    def abi_hash(self) -> str:
        """SHA-256 over the canonical ABI description (03 §1.3)."""
        return canonical_hash(self.abi_description())
