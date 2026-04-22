"""News stream ingestion for the Political Alpha agent.

Three sources, one unified event stream:
  - GDELT 2.0 DOC API  (primary, 15-min cadence, no auth)
  - CNN Truth Social archive (Trump posts, 5-min cadence)
  - RSS fallback (Reuters / Bloomberg / CNBC)

Responsibilities (in order):
  1. Fetch raw items from each enabled source.
  2. Match each item to a tracked persona by keyword.
  3. Score salience (tier × hot-term × recency).
  4. Dedupe by SHA-256 of normalized headline.
  5. Filter against a rolling SQLite cache of seen event_ids (48h TTL).

Network calls are isolated in the `_fetch_*` helpers so tests can mock them.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from core.news_types import NewsEvent, Persona, hash_headline


GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
TRUTH_SOCIAL_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_events (
    event_id TEXT PRIMARY KEY,
    seen_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seen_ts ON seen_events(seen_ts);
"""

_CACHE_TTL = timedelta(hours=48)


# ── Salience ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SalienceInputs:
    headline: str
    persona: Persona
    ts: datetime
    now: datetime


def compute_salience(inputs: SalienceInputs) -> float:
    """Blend three signals into [0, 1].

    - Tier score (persona weight): tier 1 → 0.5, tier 2 → 0.3, tier 3 → 0.15.
    - Hot-term hit: up to +0.30 if any high_salience_term is present.
    - Recency: up to +0.20 for events in the last 24h, decaying linearly.
    """
    tier_weight = {1: 0.50, 2: 0.30, 3: 0.15}[inputs.persona.tier]

    lowered = inputs.headline.lower()
    hot_hits = sum(1 for t in inputs.persona.high_salience_terms if t.lower() in lowered)
    hot_score = min(0.30, 0.15 * hot_hits)

    hours_old = max(0.0, (inputs.now - inputs.ts).total_seconds() / 3600.0)
    recency = max(0.0, 0.20 * (1.0 - min(hours_old, 24.0) / 24.0))

    return round(min(1.0, tier_weight + hot_score + recency), 4)


def gdelt_search_url(query: str, max_records: int = 50) -> str:
    return (
        f"{GDELT_BASE}?query={quote(query)}"
        f"&mode=ArtList&maxrecords={max_records}&format=json&sort=DateDesc"
    )


# ── Fetch helpers (mocked in tests) ───────────────────────────────────────────


def _fetch_gdelt(personas: Iterable[Persona]) -> list[NewsEvent]:  # pragma: no cover
    """Live GDELT fetch. Mocked in the test suite. Parallelized across
    tier-1 personas to keep UI latency bounded (~10s worst case)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = datetime.now(timezone.utc)
    targets = [p for p in personas if "gdelt" in p.watched_sources and p.tier == 1]

    def _one(p: Persona) -> list[NewsEvent]:
        query = " OR ".join(f'"{k}"' for k in p.keywords)
        url = gdelt_search_url(query=query, max_records=25)
        try:
            resp = requests.get(url, timeout=8)
            data = resp.json() if resp.ok else {}
        except Exception:
            return []
        out: list[NewsEvent] = []
        for art in data.get("articles", []) or []:
            headline = (art.get("title") or "").strip()
            if not headline:
                continue
            ts_raw = art.get("seendate", "")
            try:
                ts = datetime.strptime(ts_raw, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                ts = now
            out.append(
                NewsEvent(
                    event_id=hash_headline(headline),
                    ts=ts,
                    source="gdelt",
                    persona_id=p.id,
                    headline=headline,
                    url=art.get("url", ""),
                    raw_text=None,
                    salience=0.0,
                )
            )
        return out

    events: list[NewsEvent] = []
    if not targets:
        return events
    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as ex:
        for fut in as_completed([ex.submit(_one, p) for p in targets]):
            events.extend(fut.result())
    return events


def _fetch_truth_social(  # pragma: no cover
    personas: Iterable[Persona],
) -> list[NewsEvent]:
    """Live Truth Social archive fetch. Mocked in the test suite."""
    import requests

    trump = next((p for p in personas if p.id == "trump"), None)
    if trump is None or "truth_social" not in trump.watched_sources:
        return []
    try:
        resp = requests.get(TRUTH_SOCIAL_URL, timeout=8)
        data = resp.json() if resp.ok else []
    except Exception:
        return []

    events: list[NewsEvent] = []
    for post in (data or [])[:100]:
        text = (post.get("content") or post.get("text") or "").strip()
        if not text:
            continue
        ts_raw = post.get("created_at") or post.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        events.append(
            NewsEvent(
                event_id=hash_headline(text[:200]),
                ts=ts,
                source="truth_social",
                persona_id="trump",
                headline=text[:200],
                url=post.get("url", ""),
                raw_text=text,
                salience=0.0,
            )
        )
    return events


def _fetch_yfinance_news(tickers: Iterable[str]) -> list[NewsEvent]:  # pragma: no cover
    """Per-ticker news from Yahoo Finance. Aggregated from Reuters, Bloomberg,
    Barrons, MarketWatch, etc. — fast and reliable; mocked in tests."""
    try:
        import yfinance as yf
    except ImportError:
        return []

    events: list[NewsEvent] = []
    now = datetime.now(timezone.utc)
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).news or []
        except Exception:
            continue
        for item in items[:10]:
            content = item.get("content") or item
            title = (content.get("title") or item.get("title") or "").strip()
            if not title:
                continue
            ts_raw = (
                content.get("pubDate")
                or item.get("providerPublishTime")
                or item.get("pubDate")
            )
            ts = now
            if isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            elif isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    pass
            url_val = ""
            cp = content.get("canonicalUrl") or {}
            if isinstance(cp, dict):
                url_val = cp.get("url", "") or item.get("link", "")
            headline = f"{ticker}: {title}"
            events.append(
                NewsEvent(
                    event_id=hash_headline(headline),
                    ts=ts,
                    source="yfinance",
                    persona_id=None,
                    headline=headline,
                    url=url_val,
                    raw_text=content.get("summary") or item.get("summary"),
                    salience=0.0,
                )
            )
    return events


def _fetch_rss(personas: Iterable[Persona]) -> list[NewsEvent]:  # pragma: no cover
    """Live RSS fetch. Mocked in the test suite."""
    try:
        import feedparser
    except ImportError:
        return []

    feeds = [
        "https://feeds.reuters.com/reuters/topNews",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ]
    events: list[NewsEvent] = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue
        for entry in (parsed.entries or [])[:30]:
            headline = (entry.get("title") or "").strip()
            if not headline:
                continue
            ts_struct = entry.get("published_parsed")
            if ts_struct:
                ts = datetime(*ts_struct[:6], tzinfo=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)
            events.append(
                NewsEvent(
                    event_id=hash_headline(headline),
                    ts=ts,
                    source="rss",
                    persona_id=None,
                    headline=headline,
                    url=entry.get("link", ""),
                    raw_text=entry.get("summary"),
                    salience=0.0,
                )
            )
    return events


# ── NewsWatcher ───────────────────────────────────────────────────────────────


class NewsWatcher:
    def __init__(
        self, personas: Iterable[Persona], cache_db: str | Path
    ) -> None:
        self.personas = tuple(personas)
        self.cache_db = Path(cache_db)

    # ── cache plumbing ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_db)) as conn:
            conn.executescript(_CACHE_SCHEMA)
            self._prune(conn)
            conn.commit()

    def _prune(self, conn: sqlite3.Connection) -> None:
        cutoff = (datetime.now(timezone.utc) - _CACHE_TTL).isoformat()
        conn.execute("DELETE FROM seen_events WHERE seen_ts < ?", (cutoff,))

    def _seen_ids(self) -> set[str]:
        if not self.cache_db.exists():
            return set()
        with closing(sqlite3.connect(self.cache_db)) as conn:
            rows = conn.execute("SELECT event_id FROM seen_events").fetchall()
        return {r[0] for r in rows}

    def filter_new(self, events: Iterable[NewsEvent]) -> list[NewsEvent]:
        seen = self._seen_ids()
        # dedupe within the batch too
        by_id: dict[str, NewsEvent] = {}
        for e in events:
            if e.event_id in seen or e.event_id in by_id:
                continue
            by_id[e.event_id] = e
        return list(by_id.values())

    def mark_seen(self, events: Iterable[NewsEvent]) -> None:
        with closing(sqlite3.connect(self.cache_db)) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO seen_events (event_id, seen_ts) VALUES (?, ?)",
                [(e.event_id, datetime.now(timezone.utc).isoformat()) for e in events],
            )
            conn.commit()

    # ── persona matching + enrichment ──────────────────────────────────────

    def match_persona(self, text: str) -> str | None:
        """Pick the best-matching persona for a raw headline.

        Preference order: (tier 1 > tier 2 > tier 3), then first in list.
        """
        lowered = text.lower()
        candidates = [
            p for p in self.personas
            if any(kw.lower() in lowered for kw in p.keywords)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.tier)
        return candidates[0].id

    def _enrich(self, event: NewsEvent) -> NewsEvent:
        """Attach persona_id + salience. Returns a new (immutable) event."""
        persona_id = event.persona_id or self.match_persona(event.headline)
        persona = None
        if persona_id:
            persona = next((p for p in self.personas if p.id == persona_id), None)

        now = datetime.now(timezone.utc)
        if persona is not None:
            salience = compute_salience(
                SalienceInputs(
                    headline=event.headline,
                    persona=persona,
                    ts=event.ts,
                    now=now,
                )
            )
        elif event.source == "yfinance":
            hours_old = max(0.0, (now - event.ts).total_seconds() / 3600.0)
            recency = max(0.0, 1.0 - min(hours_old, 48.0) / 48.0)
            salience = round(0.35 + 0.25 * recency, 4)
        else:
            salience = 0.0

        return NewsEvent(
            event_id=event.event_id,
            ts=event.ts,
            source=event.source,
            persona_id=persona_id,
            headline=event.headline,
            url=event.url,
            raw_text=event.raw_text,
            salience=salience,
        )

    # ── top-level poll ─────────────────────────────────────────────────────

    def poll(
        self,
        min_salience: float = 0.0,
        portfolio_tickers: Iterable[str] = (),
    ) -> list[NewsEvent]:
        raw = (
            _fetch_gdelt(self.personas)
            + _fetch_truth_social(self.personas)
            + _fetch_rss(self.personas)
            + _fetch_yfinance_news(portfolio_tickers)
        )
        enriched = [self._enrich(e) for e in raw]
        filtered = [e for e in enriched if e.salience >= min_salience]
        return self.filter_new(filtered)
