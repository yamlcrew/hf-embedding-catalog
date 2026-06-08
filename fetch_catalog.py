#!/usr/bin/env python3
"""
HuggingFace embedding-model catalog builder.

Fetches transformers.js-compatible embedding models and stores per-model
metadata under data/<org>/<model>/model.yaml.

Catalog rules
─────────────
• Default: top 1000 models per pipeline tag by downloads (--all for no cap).
• A model is (re)fetched when its model.yaml is absent OR fetchedAt is older
  than REFRESH_DAYS (180 days).
• Fetching is INTERLEAVED with listing — files appear in data/ immediately.
• catalog.yaml is rebuilt from all existing model.yaml files at the end.

Usage
─────
    python fetch_catalog.py                    # top 1000/tag, fetch stale
    python fetch_catalog.py --all              # no cap on listing
    python fetch_catalog.py --max-list 500     # cap at 500/tag
    python fetch_catalog.py --dry-run          # list only, no writes
    python fetch_catalog.py --limit 50         # fetch at most 50 per run

Environment
───────────
    HF_TOKEN   optional (strongly recommended — avoids 429 rate limits)
"""

import argparse
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import json
import requests
import yaml

# Make stdout/stderr UTF-8 so box-drawing / em-dash / arrow characters in log
# messages never raise UnicodeEncodeError on legacy code pages (e.g. Windows
# cp1250). errors="replace" guarantees a print can never crash the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Set by Ctrl+C — workers check this instead of sleeping unconditionally.
_ABORT = threading.Event()

# Background pool for JSON → YAML conversion (fire-and-forget, never blocks fetch workers).
_YAML_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yaml")

# Global rate limiter — all threads share one token bucket so we never
# fire more than 1 request every GLOBAL_REQUEST_INTERVAL seconds.
_RATE_LOCK = threading.Lock()
_RATE_LAST: list[float] = [0.0]  # mutable so inner function can update it

# Runtime deadline (epoch seconds, 0 = no limit). A watchdog thread sets _ABORT
# when reached so the run stops gracefully, writes the catalog, and exits 0
# BEFORE the CI job is hard-killed at its timeout-minutes.
_DEADLINE: list[float] = [0.0]


def _rate_wait() -> None:
    """Block until the global rate window has elapsed, then mark the slot used."""
    with _RATE_LOCK:
        now = time.time()
        gap = GLOBAL_REQUEST_INTERVAL - (now - _RATE_LAST[0])
        if gap > 0:
            if _ABORT.wait(timeout=gap):
                raise RuntimeError("Aborted")
        _RATE_LAST[0] = time.time()


def _start_deadline_watchdog(minutes: float) -> None:
    """Arm a background timer that sets _ABORT after `minutes`, so the run
    stops gracefully and the catalog still gets written + committed."""
    if minutes <= 0:
        return
    _DEADLINE[0] = time.time() + minutes * 60.0

    def _watch() -> None:
        remaining = _DEADLINE[0] - time.time()
        # Wait until the deadline OR an earlier Ctrl+C abort wakes us.
        if remaining > 0 and _ABORT.wait(timeout=remaining):
            return  # already aborted by Ctrl+C — nothing to do
        # Set the flag FIRST so a failing print() can never block the abort.
        _ABORT.set()
        try:
            print(f"\n[catalog] runtime budget of {minutes:.0f} min reached -- "
                  f"stopping gracefully and writing catalog", flush=True)
        except Exception:
            pass

    threading.Thread(target=_watch, daemon=True, name="deadline").start()

# ── config ────────────────────────────────────────────────────────────────────

DATA_DIR          = Path("data")
CATALOG_FILE      = Path("catalog.json")
PIPELINE_TAGS     = ["sentence-similarity", "feature-extraction"]
LIBRARY           = "transformers.js"
PAGE_SIZE         = 100      # items per list page
DEFAULT_MAX_LIST  = 1000     # top-N per tag (override with --max-list / --all)
REFRESH_DAYS             = 180
GLOBAL_REQUEST_INTERVAL  = 0.1   # min seconds between ANY HTTP request across all threads (~10 req/s)
LIST_PAGE_DELAY          = 0.05  # between list pages (also subject to rate limiter)
FETCH_CONCURRENCY        = 15    # parallel model-detail workers
CATALOG_REBUILD_INTERVAL = 50    # rebuild catalog.json every N successful fetches
RETRY_BACKOFF     = [5, 15, 30, 60]
HF_API_BASE       = "https://huggingface.co/api/models"
HF_TOKEN          = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

# ── helpers ───────────────────────────────────────────────────────────────────

def hf_headers() -> dict:
    h = {"Accept": "application/json", "User-Agent": "hf-embedding-catalog/1.0"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def hf_get(url: str, timeout: int = 30) -> requests.Response:
    """GET with global rate limiting and automatic retry on 429."""
    for attempt, backoff in enumerate([0] + RETRY_BACKOFF):
        if _ABORT.is_set():
            raise RuntimeError("Aborted")
        if backoff:
            print(f"  [429] waiting {backoff}s...", flush=True)
            if _ABORT.wait(timeout=backoff):
                raise RuntimeError("Aborted")
        _rate_wait()
        try:
            resp = requests.get(url, headers=hf_headers(), timeout=timeout,
                                allow_redirects=True)
            if resp.status_code == 429:
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            if attempt == len(RETRY_BACKOFF):
                raise
    raise RuntimeError(f"Exhausted retries for {url}")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def model_path(model_id: str) -> Path:
    parts = model_id.split("/", 1)
    org, name = (parts[0], parts[1]) if len(parts) == 2 else ("_", parts[0])
    return DATA_DIR / org / name / "model.json"


def needs_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(str(data["fetchedAt"]).replace("Z", "+00:00"))
        if (datetime.now(timezone.utc) - fetched) > timedelta(days=REFRESH_DAYS):
            return True
        # Old-format: onnxFiles was a list of names, now it's {name: size_mb}.
        if isinstance(data.get("onnxFiles"), list):
            return True
        # Sizes missing: fetched before tree-API support — all onnx sizes are 0.
        onnx = data.get("onnxFiles")
        if isinstance(onnx, dict) and onnx and all(v == 0 for v in onnx.values()):
            return True
        return False
    except Exception:
        return True


# ── HF fetch ──────────────────────────────────────────────────────────────────

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "I4": 0.5}


def fetch_model_detail(model_id: str, fetched_at: str) -> dict:
    """
    Fetch full per-model metadata and return as-is from the HF API.

    Requests:
      1. GET /api/models/<id>?full=true                  — complete base record
      2. GET /api/models/<id>?expand=safetensors&...     — extra expand fields merged in
      3. GET /api/models/<id>/tree/main?recursive=true   — file tree with real sizes
      4. GET <id>/resolve/main/config.json               — stored verbatim as "configJson"

    Added keys (not from HF, computed locally):
      onnxFiles       — {path → size_mb} for every .onnx file
      safetensorsSizeMb — sum of all .safetensors file sizes (decimal MB)
      approxSizeMb    — params × bytes/dtype / 1_000_000
      fetchedAt       — ISO timestamp of this fetch
    """

    # 1) complete raw base record
    data: dict = hf_get(f"{HF_API_BASE}/{model_id}?full=true", timeout=30).json()

    # 2) extra expand fields — merged into data
    try:
        extra = hf_get(
            f"{HF_API_BASE}/{model_id}"
            "?expand=safetensors"
            "&expand=cardData"
            "&expand=downloadsAllTime"
            "&expand=inferenceProviderMapping"
            "&expand=transformersInfo",
            timeout=20,
        ).json()
        for key in ("safetensors", "cardData", "downloadsAllTime",
                    "inferenceProviderMapping", "transformersInfo"):
            if key in extra:
                data[key] = extra[key]
    except Exception:
        pass

    # 3) file tree with real byte sizes (lfs.size for LFS objects, size for regular)
    try:
        tree: list = hf_get(
            f"https://huggingface.co/api/models/{model_id}/tree/main?recursive=true",
            timeout=30,
        ).json()
        size_map: dict[str, int] = {
            item["path"]: (item.get("lfs") or {}).get("size") or item.get("size") or 0
            for item in tree
            if isinstance(item, dict) and item.get("type") == "file"
        }
    except Exception:
        size_map = {}

    # 4) full config.json stored verbatim
    try:
        data["configJson"] = hf_get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            timeout=15,
        ).json()
    except Exception:
        data.setdefault("configJson", None)

    # ── computed helpers ──────────────────────────────────────────────────────

    def _size_mb(path: str) -> float:
        return round(size_map.get(path, 0) / 1_000_000, 1)

    # onnxFiles: {path → size_mb} for every .onnx file in the repo
    data["onnxFiles"] = {
        path: _size_mb(path)
        for path in size_map
        if path.endswith(".onnx")
    }

    # safetensorsSizeMb: sum of all .safetensors file sizes (decimal MB)
    data["safetensorsSizeMb"] = round(
        sum(size_map[p] for p in size_map if p.endswith(".safetensors")) / 1_000_000, 1
    )

    # approxSizeMb: params × bytes/dtype in decimal MB (fallback when tree unavailable)
    params_by_dtype = (data.get("safetensors") or {}).get("parameters") or {}
    data["approxSizeMb"] = round(
        sum(count * _DTYPE_BYTES.get(dtype, 4) for dtype, count in params_by_dtype.items())
        / 1_000_000,
        1,
    )

    data["fetchedAt"] = fetched_at
    return data


# ── catalog ───────────────────────────────────────────────────────────────────

def _cfg(d: dict, *keys, default=0):
    """Read a value from configJson with fallback keys."""
    cfg = d.get("configJson") or {}
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            try:
                return type(default)(v)
            except Exception:
                return default
    # also check top-level for old-format model.json files
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return type(default)(v)
            except Exception:
                return default
    return default


def _extract_lang(d: dict) -> tuple[str, bool]:
    """Return (lang_string, is_multilingual).

    lang_string: deduplicated ISO 639-1 language codes, comma-separated.
    is_multilingual: True when "multilingual" appears in cardData.language or tags.

    "multilingual" is NOT included in lang_string — it becomes the separate boolean.
    Sources (merged, deduplicated):
      1. cardData.language  — authoritative list from the model card YAML
      2. tags               — 2-letter lowercase alpha codes only (ISO 639-1 shape)
    """
    langs: set[str] = set()
    multilingual = False

    # 1) cardData.language — can be a list or a single string
    raw = (d.get("cardData") or {}).get("language") or []
    if isinstance(raw, str):
        raw = [raw]
    for item in raw:
        s = str(item).strip().lower()
        if s == "multilingual":
            multilingual = True
        elif s:
            langs.add(s)

    # 2) tags — only accept exactly 2 lowercase alpha chars; detect "multilingual"
    for tag in (d.get("tags") or []):
        if not isinstance(tag, str):
            continue
        t = tag.strip().lower()
        if t == "multilingual":
            multilingual = True
        elif len(t) == 2 and t.isalpha():
            langs.add(t)

    return ",".join(sorted(langs)), multilingual


def build_catalog() -> list[dict]:
    entries = []
    for p in DATA_DIR.rglob("model.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            st = d.get("safetensors") or {}
            params_by_dtype = st.get("parameters") or {}
            param_count = st.get("total") or sum(params_by_dtype.values()) or d.get("paramCount", 0)
            cfg = d.get("configJson") or {}

            # model_size_mb — actual single-file download size (decimal MB, matches HF display)
            # priority: safetensorsSizeMb from siblings > approxSizeMb from params
            model_size_mb = (
                d.get("safetensorsSizeMb")
                or d.get("approxSizeMb")
                or round(
                    sum(c * _DTYPE_BYTES.get(t, 4) for t, c in params_by_dtype.items())
                    / 1_000_000, 1
                )
            )

            # onnx_size_mb — size of the main onnx/model.onnx only (decimal MB)
            raw_onnx = d.get("onnxFiles") or {}
            if isinstance(raw_onnx, dict):
                onnx_size_mb = raw_onnx.get("onnx/model.onnx") or raw_onnx.get("model.onnx") or 0
            else:
                onnx_size_mb = 0

            lang, multilingual = _extract_lang(d)

            entries.append({
                "id":             d.get("id", ""),
                "pipeline_tag":   d.get("pipeline_tag") or d.get("pipelineTag", ""),
                "library_name":   d.get("library_name") or d.get("libraryName", ""),
                "author":         d.get("author", ""),
                "model_type":     cfg.get("model_type") or d.get("modelType") or d.get("model_type", ""),
                "architectures":  (cfg.get("architectures") or [""])[0] if cfg.get("architectures") else d.get("architectures", ""),
                "hidden_size":    _cfg(d, "hidden_size", "n_embd", "d_model"),
                "max_position_embeddings": _cfg(d, "max_position_embeddings", "n_positions", "n_ctx"),
                "num_hidden_layers": _cfg(d, "num_hidden_layers", "n_layer"),
                "torch_dtype":    cfg.get("torch_dtype") or d.get("torchDtype") or d.get("torch_dtype", ""),
                "param_count":    int(param_count),
                "model_size_mb":  model_size_mb,
                "onnx_size_mb":   onnx_size_mb,
                "downloads":      int(d.get("downloads", 0)),
                "likes":          int(d.get("likes", 0)),
                "trending_score": float(d.get("trendingScore") or d.get("trending_score") or 0),
                "has_onnx":       bool(onnx_size_mb or raw_onnx),
                "has_safetensors": bool(d.get("safetensorsSizeMb") or any(
                    isinstance(s, dict) and s.get("rfilename", "").endswith(".safetensors")
                    for s in (d.get("siblings") or [])
                ) or d.get("hasSafetensors")),
                "lang":           lang,
                "multilingual":   multilingual,
                "gated":          bool(d.get("gated")),
                "created_at":     d.get("createdAt") or d.get("created_at", ""),
                "last_modified":  d.get("lastModified") or d.get("last_modified", ""),
                "fetched_at":     d.get("fetchedAt") or d.get("fetched_at", ""),
            })
        except Exception:
            pass
    entries.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    return entries


# ── catalog write ────────────────────────────────────────────────────────────

def _write_catalog() -> int:
    catalog = build_catalog()
    CATALOG_FILE.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in catalog),
        encoding="utf-8",
    )
    return len(catalog)


# ── listing (streamed into fetch) ─────────────────────────────────────────────

def json2yaml(json_path: Path) -> None:
    """Convert a model.json to model.yaml in-place (background task)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.with_suffix(".yaml").write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120),
        encoding="utf-8",
    )


def _fetch_and_save(mid: str, fetched_at: str) -> dict:
    """Worker: fetch full model detail and write model.json. Returns detail dict."""
    detail = fetch_model_detail(mid, fetched_at)
    path = model_path(mid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
    _YAML_POOL.submit(json2yaml, path)
    return detail


def process_tag(tag: str, max_models: int, seen: set, fetched_at: str,
                limit: int, total_fetched: int, dry_run: bool) -> tuple[int, int, list[str]]:
    """
    Phase 1: list all pages for one tag (fast, sequential).
    Phase 2: fetch stale models in parallel (FETCH_CONCURRENCY workers).
    Returns (fetched, errors, new_ids).
    """
    # ── Phase 1: listing ──────────────────────────────────────────────────────
    url = (
        f"{HF_API_BASE}?library={LIBRARY}"
        f"&pipeline_tag={tag}"
        f"&sort=downloads&direction=-1"
        f"&limit={PAGE_SIZE}"
        f"&full=true"
    )
    listed = 0
    page = 0
    new_ids: list[str] = []

    while url:
        if _ABORT.is_set():
            print(f"  [{tag}] listing stopped at {listed} models (deadline/abort)", flush=True)
            break
        resp = hf_get(url, timeout=30)
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        for m in batch:
            mid = m.get("id") or m.get("modelId")
            if mid and mid not in seen:
                seen.add(mid)
                new_ids.append(mid)
            listed += 1
            if max_models and listed >= max_models:
                break
        link = resp.headers.get("link", "")
        url = None
        if link and (not max_models or listed < max_models):
            m2 = re.search(r'<([^>]+)>;\s*rel="next"', link)
            if m2:
                url = m2.group(1)
        page += 1
        cap_str = f"/{max_models}" if max_models else ""
        print(f"  [{tag}] page {page}: {len(batch)} listed  (total {listed}{cap_str})", flush=True)
        if url:
            time.sleep(LIST_PAGE_DELAY)

    print(f"  [{tag}] listing done: {len(new_ids)} new unique models", flush=True)

    # ── Phase 2: parallel fetch of stale models ───────────────────────────────
    stale = []
    for mid in new_ids:
        path = model_path(mid)
        if needs_refresh(path):
            stale.append(mid)
        else:
            print(f"  skip  {mid}  (cached)", flush=True)

    if not stale:
        print(f"  [{tag}] all {len(new_ids)} models up-to-date, nothing to fetch")
        return 0, 0, new_ids

    remaining = (limit - total_fetched) if limit > 0 else len(stale)
    to_fetch = stale[:max(0, remaining)]

    if dry_run:
        print(f"  [{tag}] would fetch {len(to_fetch)} stale / {len(new_ids) - len(stale)} cached")
        return 0, 0, new_ids

    print(f"  [{tag}] {len(new_ids) - len(stale)} cached, fetching {len(to_fetch)} stale "
          f"({FETCH_CONCURRENCY} parallel)...", flush=True)

    fetched = errors = 0
    done_count = 0
    aborted = False
    pool = ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY)
    futures = {pool.submit(_fetch_and_save, mid, fetched_at): mid for mid in to_fetch}
    try:
        for future in as_completed(futures):
            if _ABORT.is_set():           # deadline reached — stop reading results
                aborted = True
                break
            mid = futures[future]
            done_count += 1
            try:
                detail = future.result()
                fetched += 1
                cfg = (detail.get("configJson") or {})
                print(f"  [{done_count}/{len(to_fetch)}] ok  {mid}"
                      f"  dim={cfg.get('hidden_size', '?')} type={cfg.get('model_type', '')}",
                      flush=True)
                if fetched % CATALOG_REBUILD_INTERVAL == 0:
                    _write_catalog()
                    print(f"  [catalog] rebuilt after {fetched} fetches", flush=True)
            except Exception as e:
                if _ABORT.is_set():       # in-flight worker raised "Aborted" — not a real error
                    aborted = True
                    break
                errors += 1
                print(f"  [{done_count}/{len(to_fetch)}] ERROR {mid}: {e}",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[catalog] Ctrl+C — cancelling pending fetches...", flush=True)
        _ABORT.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise

    if aborted or _ABORT.is_set():
        print(f"  [{tag}] stopped early — {fetched} fetched before deadline", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
    else:
        pool.shutdown(wait=True)

    return fetched, errors, new_ids


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build HF embedding-model catalog")
    parser.add_argument("--dry-run",   action="store_true",
                        help="List models only, do not write any files")
    parser.add_argument("--all",       action="store_true",
                        help="Fetch ALL models (no cap on listing)")
    parser.add_argument("--max-list",  type=int, default=DEFAULT_MAX_LIST,
                        help=f"Max models to list per tag (default {DEFAULT_MAX_LIST})")
    parser.add_argument("--limit",     type=int, default=0,
                        help="Max stale models to (re)fetch per run (default: all stale)")
    parser.add_argument("--max-runtime-min", type=float,
                        default=float(os.environ.get("MAX_RUNTIME_MIN", "0")),
                        help="Stop gracefully after N minutes, write catalog, exit 0 "
                             "(0 = no limit; env: MAX_RUNTIME_MIN)")
    args = parser.parse_args()

    max_list = 0 if args.all else args.max_list

    if args.max_runtime_min > 0:
        print(f"[catalog] Runtime budget: {args.max_runtime_min:.0f} min "
              f"(will stop + write catalog before then)")
        _start_deadline_watchdog(args.max_runtime_min)

    if HF_TOKEN:
        print(f"[catalog] HF_TOKEN present ({HF_TOKEN[:8]}...)")
    else:
        print("[catalog] No HF_TOKEN — anonymous rate limits apply (use HF_TOKEN)")

    fetched_at = now_iso()
    seen: set[str] = set()
    total_fetched = total_errors = 0

    try:
        for tag in PIPELINE_TAGS:
            if _ABORT.is_set():
                break
            print(f"\n[catalog] === pipeline_tag={tag}"
                  + (f" (top {max_list})" if max_list else " (all)") + " ===")
            f, e, ids = process_tag(
                tag, max_list, seen, fetched_at,
                args.limit, total_fetched, args.dry_run
            )
            total_fetched += f
            total_errors  += e
            print(f"  [{tag}] done: {len(ids)} new models listed, "
                  f"{f} fetched, {e} errors")

            if _ABORT.is_set():
                print(f"\n[catalog] runtime budget reached — stopping after {tag}", flush=True)
                break
            if args.limit > 0 and total_fetched >= args.limit:
                print(f"\n[catalog] --limit {args.limit} reached, stopping")
                break
    except KeyboardInterrupt:
        print(f"\n[catalog] Interrupted after {total_fetched} fetches — building partial catalog",
              flush=True)

    # ── rebuild catalog.json → catalog.yaml ──────────────────────────────────
    print(f"\n[catalog] Summary: {total_fetched} fetched, {total_errors} errors")
    print("[catalog] Rebuilding catalog.json...")
    n = _write_catalog()
    print(f"[catalog] Converting catalog.json → catalog.yaml...")
    catalog_data = [json.loads(line) for line in CATALOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    Path("catalog.yaml").write_text(
        yaml.dump(catalog_data, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120),
        encoding="utf-8",
    )
    print(f"[catalog] Done: {n} models")


if __name__ == "__main__":
    main()
