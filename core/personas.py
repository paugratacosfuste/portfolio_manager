"""YAML → tuple[Persona, ...] loader.

Keeps persona config human-editable (data/personas.yaml) while the rest
of the pipeline consumes immutable typed values. Validation errors at
load time are loud: a duplicate id or unknown source fails fast instead
of polluting the ledger later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.news_types import AssetImpact, Persona


def load_personas(path: str | Path) -> tuple[Persona, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top level must be a list of persona dicts")

    seen: set[str] = set()
    personas: list[Persona] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {idx} is not a mapping")

        pid = entry.get("id")
        if not pid:
            raise ValueError(f"{path}: entry {idx} missing id")
        if pid in seen:
            raise ValueError(f"{path}: duplicate persona id {pid!r}")
        seen.add(pid)

        personas.append(_persona_from_dict(entry))

    return tuple(personas)


def _persona_from_dict(entry: dict[str, Any]) -> Persona:
    impacts = tuple(
        AssetImpact(
            asset=ai["asset"],
            direction_on_positive=ai["direction_on_positive"],
        )
        for ai in entry.get("primary_asset_impacts", [])
    )
    return Persona(
        id=entry["id"],
        name=entry["name"],
        role=entry["role"],
        tier=int(entry["tier"]),
        watched_sources=tuple(entry["watched_sources"]),
        keywords=tuple(entry["keywords"]),
        high_salience_terms=tuple(entry.get("high_salience_terms", [])),
        primary_asset_impacts=impacts,
        historical_reaction_horizon_hours=int(
            entry["historical_reaction_horizon_hours"]
        ),
    )
