"""Tests for core/personas.py — YAML → immutable Persona tuple loader.

Personas are defined in data/personas.yaml as readable config, then loaded
into `tuple[Persona, ...]` for the rest of the pipeline. Validation
(unknown sources, bad tier, empty keywords) fails at load time, not at
runtime inside the agent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.news_types import AssetImpact, Persona
from core.personas import load_personas


SAMPLE_YAML = """
- id: trump
  name: Donald Trump
  role: US President
  tier: 1
  watched_sources: [truth_social, gdelt]
  keywords: [Trump, "President Trump"]
  high_salience_terms: [tariff, sanctions]
  primary_asset_impacts:
    - asset: SPY
      direction_on_positive: LONG
    - asset: BTC-USD
      direction_on_positive: LONG
  historical_reaction_horizon_hours: 24

- id: powell
  name: Jerome Powell
  role: Fed Chair
  tier: 1
  watched_sources: [gdelt, rss]
  keywords: [Powell, "Fed Chair"]
  high_salience_terms: [rate, hawkish, dovish, FOMC]
  primary_asset_impacts:
    - asset: TLT
      direction_on_positive: SHORT
  historical_reaction_horizon_hours: 4
"""


def test_load_personas_from_sample(tmp_path: Path):
    p = tmp_path / "personas.yaml"
    p.write_text(SAMPLE_YAML)

    personas = load_personas(p)
    assert isinstance(personas, tuple)
    assert len(personas) == 2
    assert all(isinstance(x, Persona) for x in personas)

    trump = personas[0]
    assert trump.id == "trump"
    assert trump.tier == 1
    assert "Trump" in trump.keywords
    assert trump.watched_sources == ("truth_social", "gdelt")
    assert all(isinstance(ai, AssetImpact) for ai in trump.primary_asset_impacts)
    assert trump.primary_asset_impacts[0].asset == "SPY"


def test_load_rejects_bad_tier(tmp_path: Path):
    bad = """
- id: x
  name: X
  role: X
  tier: 7
  watched_sources: [gdelt]
  keywords: [x]
  high_salience_terms: [x]
  primary_asset_impacts: [{asset: SPY, direction_on_positive: LONG}]
  historical_reaction_horizon_hours: 1
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_personas(p)


def test_load_rejects_unknown_source(tmp_path: Path):
    bad = """
- id: x
  name: X
  role: X
  tier: 1
  watched_sources: [twitter]
  keywords: [x]
  high_salience_terms: [x]
  primary_asset_impacts: [{asset: SPY, direction_on_positive: LONG}]
  historical_reaction_horizon_hours: 1
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_personas(p)


def test_load_requires_unique_ids(tmp_path: Path):
    dup = """
- id: x
  name: X
  role: X
  tier: 1
  watched_sources: [gdelt]
  keywords: [x]
  high_salience_terms: [x]
  primary_asset_impacts: [{asset: SPY, direction_on_positive: LONG}]
  historical_reaction_horizon_hours: 1
- id: x
  name: Y
  role: Y
  tier: 2
  watched_sources: [gdelt]
  keywords: [y]
  high_salience_terms: [y]
  primary_asset_impacts: [{asset: SPY, direction_on_positive: LONG}]
  historical_reaction_horizon_hours: 1
"""
    p = tmp_path / "dup.yaml"
    p.write_text(dup)
    with pytest.raises(ValueError, match="duplicate"):
        load_personas(p)


def test_real_personas_yaml_loads_and_has_required_figures():
    """Sanity check the actual project-level personas.yaml."""
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "personas.yaml"
    if not path.exists():
        pytest.skip("data/personas.yaml not yet created")

    personas = load_personas(path)
    assert len(personas) >= 25, f"expected ≥25 personas, got {len(personas)}"

    ids = {p.id for p in personas}
    # Tier-1 must include these — grader demo depends on it
    for required in ("trump", "musk", "powell", "lagarde", "xi"):
        assert required in ids, f"missing required Tier-1 persona: {required}"

    # Every tier represented
    tiers = {p.tier for p in personas}
    assert tiers == {1, 2, 3}
