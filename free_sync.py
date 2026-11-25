#!/usr/bin/env python3
"""Sync live-verified free chat models into 9Router's `free-burst` combo.

Every candidate is harvested from 9Router's `/v1/models` (plus OpenRouter's free
catalog prefixed with `openrouter/`) and must pass a live gateway probe using
streaming, tools, and `reasoning_effort` before entering `free-burst`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request

DB_PATH = os.environ.get(
    "NINEROUTER_DB_PATH",
    os.path.expanduser("~/.9router/db/data.sqlite"),
)
COMBO_NAME = os.environ.get("NINEROUTER_FREE_COMBO", "free-burst")
GATEWAY_URL = os.environ.get("NINEROUTER_URL", "http://localhost:20128").rstrip("/")
MAX_PER_PROVIDER = 2
PROBE_TIMEOUT_S = 8.0
PROVIDER_PROBE_DELAY_S = 0.4

GROQ_FREE_IDS = frozenset(
    {
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/mixtral-8x7b-32768",
        "groq/gemma2-9b-it",
        "groq/llama-guard-4-12b",
        "groq/qwen/qwen3-32b",
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
    }
)

# Non-chat / specialized endpoints that must never enter a chat+tools combo.
NON_CHAT_TOKENS = (
    "image",
    "tts",
    "embed",
    "pii",
    "starcoder",
    "guard",
    "clef",
    "audio",
    "video",
    "moderation",
    "rerank",
)

PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Connectivity check",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]

CREATE_EXCLUDED_TABLE = """
CREATE TABLE IF NOT EXISTS excluded_models (
    model_id TEXT PRIMARY KEY,
    provider TEXT,
    excluded_at TEXT,
    last_probe_at TEXT,
    last_probe_status TEXT
)
"""


def _gateway_key() -> str | None:
    """Read NINEROUTER_KEY from env, falling back to ~/.hermes/.env (cron context)."""
    key = os.environ.get("NINEROUTER_KEY")
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.hermes/.env"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("NINEROUTER_KEY="):
                    return (
                        line.strip()
                        .split("=", 1)[1]
                        .strip()
                        .strip('"')
                        .strip("'")
                    )
    except OSError:
        return None
    return None


def get_active_providers(conn: sqlite3.Connection) -> set[str]:
    """Provider names with at least one active connection in 9Router."""
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT provider FROM providerConnections WHERE isActive=1")
    return {row[0] for row in cur.fetchall()}


def fetch_9router_candidates(key: str) -> list[tuple[str, str]]:
    """Harvest free/chat-capable candidates from 9Router's `/v1/models`."""
    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in data.get("data", []):
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get("id") or "")
        owned_by = str(entry.get("owned_by") or mid.split("/")[0])
        if not mid or owned_by == "combo" or mid in seen:
            continue
        low = mid.lower()
        if any(tok in low for tok in NON_CHAT_TOKENS):
            continue
        caps = entry.get("capabilities")
        if isinstance(caps, dict) and caps.get("tools") is False:
            continue
        is_candidate = (
            ":free" in low
            or "-free" in low
            or "/free" in low
            or mid in GROQ_FREE_IDS
            or owned_by in ("gemini", "nvidia", "mmf")
            or (owned_by == "ollama" and ":cloud" in low)
        )
        if is_candidate:
            seen.add(mid)
            candidates.append((owned_by, mid))
    return candidates


def fetch_openrouter_candidates() -> list[tuple[str, str]]:
    """Fetch OpenRouter zero-cost models prefixed as `openrouter/<id>`."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "9router-free-sync/2.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)

    out: list[tuple[str, str]] = []
    for entry in data.get("data", []):
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get("id") or "")
        if not mid or mid == "openrouter/free":
            continue
        low = mid.lower()
        if any(tok in low for tok in NON_CHAT_TOKENS):
            continue
        pricing = entry.get("pricing") or {}
        try:
            prompt_cost = float(pricing.get("prompt", "1"))
            comp_cost = float(pricing.get("completion", "1"))
        except (TypeError, ValueError):
            continue
        if prompt_cost == 0 and comp_cost == 0:
            out.append(("openrouter", f"openrouter/{mid}"))
    return sorted(out)


def probe_gateway(
    model_id: str,
    key: str,
    timeout: float = PROBE_TIMEOUT_S,
) -> tuple[bool, float, str]:
    """Probe one route with streaming + tools + reasoning_effort.

    Returns `(ok, latency_s, reason)`.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "tools": PROBE_TOOLS,
            "reasoning_effort": "medium",
            "stream": True,
            "max_tokens": 24,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False, time.monotonic() - t0, f"http_{resp.status}"
            chunk = resp.read(2048).decode("utf-8", errors="replace")
        dt = time.monotonic() - t0
        if "data:" not in chunk or '"choices"' not in chunk:
            return False, dt, "bad_sse_payload"
        return True, dt, "ok"
    except urllib.error.HTTPError as exc:
        dt = time.monotonic() - t0
        return False, dt, f"http_{exc.code}"
    except Exception as exc:
        dt = time.monotonic() - t0
        return False, dt, type(exc).__name__


def probe_by_provider(
    candidates: list[tuple[str, str]],
    key: str,
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Probe providers concurrently while pacing requests within each provider."""
    grouped: dict[str, list[str]] = {}
    for owner, mid in candidates:
        grouped.setdefault(owner, []).append(mid)

    passing: dict[str, float] = {}
    dropped: list[tuple[str, str]] = []

    def _run_group(models: list[str]) -> tuple[dict[str, float], list[tuple[str, str]]]:
        ok_map: dict[str, float] = {}
        fail_list: list[tuple[str, str]] = []
        for idx, mid in enumerate(models):
            if idx > 0:
                time.sleep(PROVIDER_PROBE_DELAY_S)
            ok, dt, reason = probe_gateway(mid, key)
            if ok:
                ok_map[mid] = dt
            else:
                fail_list.append((mid, reason))
        return ok_map, fail_list

    if not grouped:
        return passing, dropped

    with ThreadPoolExecutor(max_workers=min(8, len(grouped))) as pool:
        futures = [pool.submit(_run_group, mids) for mids in grouped.values()]
        for fut in as_completed(futures):
            ok_map, fail_list = fut.result()
            passing.update(ok_map)
            dropped.extend(fail_list)

    return passing, dropped


def cap_per_provider(
    passing_latencies: dict[str, float],
    max_per: int = MAX_PER_PROVIDER,
) -> list[str]:
    """Keep up to `max_per` fastest verified models per provider prefix."""
    by_prov: dict[str, list[tuple[float, str]]] = {}
    for mid, dt in passing_latencies.items():
        prov = mid.split("/")[0] if "/" in mid else mid
        by_prov.setdefault(prov, []).append((dt, mid))
    kept: list[str] = []
    for entries in by_prov.values():
        entries.sort(key=lambda item: (item[0], item[1]))
        kept.extend(mid for _, mid in entries[:max_per])
    return sorted(kept)


def sync_excluded_table(
    conn: sqlite3.Connection,
    passing: set[str],
    dropped: list[tuple[str, str]],
) -> None:
    """Record failed probe reasons and clear any model that recovered."""
    cur = conn.cursor()
    cur.execute(CREATE_EXCLUDED_TABLE)
    now_iso = datetime.now(timezone.utc).isoformat()
    for mid in passing:
        cur.execute("DELETE FROM excluded_models WHERE model_id=?", (mid,))
    for mid, reason in dropped:
        prov = mid.split("/")[0] if "/" in mid else mid
        cur.execute(
            """
            INSERT OR REPLACE INTO excluded_models
                (model_id, provider, excluded_at, last_probe_at, last_probe_status)
            VALUES (?, ?, COALESCE((SELECT excluded_at FROM excluded_models WHERE model_id=?), ?), ?, ?)
            """,
            (mid, prov, mid, now_iso, now_iso, reason),
        )
    conn.commit()


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    key = _gateway_key()
    if not key:
        print("FAIL: NINEROUTER_KEY unavailable")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT models FROM combos WHERE name=?", (COMBO_NAME,))
    row = cur.fetchone()
    if not row:
        conn.close()
        print(f"FAIL: Combo '{COMBO_NAME}' not found")
        sys.exit(1)

    current_combo = json.loads(row[0])
    current_set = set(current_combo)
    active = get_active_providers(conn)

    try:
        candidates = fetch_9router_candidates(key)
    except Exception as exc:
        conn.close()
        print(f"FAIL: could not read 9Router /v1/models: {exc}")
        sys.exit(1)

    seen_ids = {mid for _, mid in candidates}
    if "openrouter" in active:
        try:
            for owner, mid in fetch_openrouter_candidates():
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    candidates.append((owner, mid))
        except Exception as exc:
            print(f"WARN: OpenRouter fetch failed: {exc}")

    passing_latencies, dropped = probe_by_provider(candidates, key)
    sync_excluded_table(conn, set(passing_latencies), dropped)

    if not passing_latencies:
        conn.close()
        print(
            f"WARN [{now}]: 0/{len(candidates)} candidates passed probe; keeping existing '{COMBO_NAME}' unchanged"
        )
        sys.exit(1)

    final_models = cap_per_provider(passing_latencies)
    final_set = set(final_models)
    new_additions = sorted(final_set - current_set)
    removed = sorted(current_set - final_set)

    if not new_additions and not removed:
        conn.close()
        return

    cur.execute(
        "UPDATE combos SET models=?, updatedAt=? WHERE name=?",
        (
            json.dumps(final_models),
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            COMBO_NAME,
        ),
    )
    conn.commit()
    conn.close()

    parts = [
        f"[{now}] Updated '{COMBO_NAME}': {len(final_models)} verified free models "
        f"(probed {len(candidates)}, {len(passing_latencies)} passed)"
    ]
    if new_additions:
        parts.append(
            "  + Added: "
            + ", ".join(f"{m} ({passing_latencies[m]:.2f}s)" for m in new_additions)
        )
    if removed:
        parts.append(f"  - Removed: {', '.join(removed)}")
    prov_counts: dict[str, int] = {}
    for mid in final_models:
        prov = mid.split("/")[0]
        prov_counts[prov] = prov_counts.get(prov, 0) + 1
    parts.append(
        "  = Per-provider: "
        + ", ".join(f"{p}:{c}" for p, c in sorted(prov_counts.items()))
    )
    print("\n".join(parts))


if __name__ == "__main__":
    main()
