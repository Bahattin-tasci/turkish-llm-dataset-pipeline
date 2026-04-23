"""
turkish_dataset_pipeline.py
BellaTurca + Cosmos + FineWeb2 → exact dedup (SHA-256+SQLite) + near-dedup (MinHash) → HuggingFace

Requirements:
    pip install datasets huggingface_hub datasketch tqdm

Usage:
    python turkish_dataset_pipeline.py run_stream

Note: Run `huggingface-cli login` or set HF_TOKEN environment variable before use.
"""

import os
import io
import sys
import json
import time
import hashlib
import sqlite3
import signal
from tqdm import tqdm

from datasets import load_dataset
from huggingface_hub import HfApi
from datasketch import MinHash, MinHashLSH


# ===============================================================
# CONFIGURATION
# ===============================================================

HF_REPO       = "tascib/turkish-llm-dataset"
PRIVATE_REPO  = True
SHARD_SIZE    = 50_000
LSH_PERM      = 64
LSH_THRESHOLD = 0.85

SQLITE_PATH   = os.path.abspath("exact_hashes.sqlite")
PROGRESS_PATH = os.path.abspath("pipeline_progress.json")

MIN_CHARS = 20
MAX_CHARS = 500_000


# ===============================================================
# GRACEFUL SHUTDOWN
# ===============================================================

stop_requested = False

def _signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("\nStopping... current shard will be completed.")

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ===============================================================
# HF TOKEN
# ===============================================================

def _read_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return None


# ===============================================================
# EXACT DEDUPLICATION — SHA-256 + SQLITE
# ===============================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_sqlite(path: str = SQLITE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("CREATE TABLE IF NOT EXISTS hashes(hash TEXT PRIMARY KEY)")
    conn.commit()
    return conn


_pending_hashes: list[str] = []
_BATCH_SIZE = 2000

def is_new_and_mark(conn: sqlite3.Connection, hash_value: str) -> bool:
    global _pending_hashes
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM hashes WHERE hash=?", (hash_value,))
    if cur.fetchone():
        return False
    _pending_hashes.append(hash_value)
    if len(_pending_hashes) >= _BATCH_SIZE:
        conn.executemany("INSERT OR IGNORE INTO hashes(hash) VALUES(?)",
                         [(h,) for h in _pending_hashes])
        conn.commit()
        _pending_hashes = []
    return True


def flush_sqlite(conn: sqlite3.Connection) -> None:
    global _pending_hashes
    if _pending_hashes:
        conn.executemany("INSERT OR IGNORE INTO hashes(hash) VALUES(?)",
                         [(h,) for h in _pending_hashes])
        conn.commit()
        _pending_hashes = []


# ===============================================================
# NEAR-DUPLICATE DETECTION — MINHASH LSH
# ===============================================================

def make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=LSH_PERM)
    for word in text.split():
        m.update(word.encode("utf-8"))
    return m


def is_near_duplicate(lsh: MinHashLSH, minhash: MinHash) -> bool:
    return len(lsh.query(minhash)) > 0


# ===============================================================
# RESUME / PROGRESS
# ===============================================================

def load_progress() -> dict:
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH) as f:
                data = json.load(f)
            print(f"Checkpoint found: source={data.get('source', '?')}, "
                  f"shard_id={data.get('shard_id', 0)}, "
                  f"records_processed={data.get('records_processed', 0)}")
            return data
        except Exception:
            pass
    return {"source": None, "shard_id": 0, "records_processed": 0}


def save_progress(source: str, shard_id: int, records_processed: int) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "source"           : source,
            "shard_id"         : shard_id,
            "records_processed": records_processed,
        }, f)


def clear_progress() -> None:
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
        print("Checkpoint cleared.")


# ===============================================================
# HUGGINGFACE HELPERS
# ===============================================================

def create_hf_repo(repo_id: str, private: bool = PRIVATE_REPO) -> HfApi:
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"create_repo warning: {e}")
    return api


def upload_bytes_to_hf(api: HfApi, repo_id: str, path_in_repo: str,
                       data: io.BytesIO, max_retries: int = 10) -> bool:
    data.seek(0)
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=data,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            return True
        except Exception as e:
            wait = min(2 ** attempt, 300)
            print(f"Upload attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return False


def flush_shard(api: HfApi, repo_id: str, shard: list,
                shard_id: int, prefix: str) -> None:
    content      = "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in shard) + "\n"
    buf          = io.BytesIO(content.encode("utf-8"))
    path_in_repo = f"{prefix}_{shard_id}.jsonl"
    print(f"\n  -> Uploading {path_in_repo}  ({len(shard):,} records)")
    upload_bytes_to_hf(api, repo_id, path_in_repo, buf)


# ===============================================================
# CORE PROCESSOR
# ===============================================================

def process_source(
    source_name   : str,
    dataset_iter,
    api           : HfApi,
    conn          : sqlite3.Connection,
    start_shard_id: int = 0,
    start_records : int = 0,
    shard_prefix  : str = "data/dataset_part",
) -> int:
    shard        = []
    shard_id     = start_shard_id
    local_idx    = 0
    records_seen = 0
    lsh          = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_PERM)

    if start_records > 0:
        print(f"  Resuming: skipping {start_records:,} records...")

    for row in tqdm(dataset_iter, desc=f"  {source_name}", unit="rec"):
        if stop_requested:
            print("  Stop signal received.")
            save_progress(source_name, shard_id, records_seen)
            break

        records_seen += 1

        if records_seen <= start_records:
            continue

        text = row.get("text") or row.get("content") or row.get("sentence") or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()

        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue

        h = sha256_text(text)
        if not is_new_and_mark(conn, h):
            continue

        mh = make_minhash(text)
        if is_near_duplicate(lsh, mh):
            continue

        lsh.insert(f"{shard_id}_{local_idx}", mh)
        local_idx += 1
        shard.append(text)

        if len(shard) >= SHARD_SIZE:
            flush_shard(api, HF_REPO, shard, shard_id, shard_prefix)
            shard     = []
            shard_id += 1
            local_idx = 0
            save_progress(source_name, shard_id, records_seen)
            lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_PERM)

    shards_produced = shard_id - start_shard_id
    if shard:
        flush_shard(api, HF_REPO, shard, shard_id, shard_prefix)
        shards_produced += 1
        save_progress(source_name, shard_id + 1, records_seen)

    flush_sqlite(conn)
    return shards_produced


# ===============================================================
# DATA SOURCE STREAM GENERATORS
# ===============================================================

def stream_bellaturca():
    dataset_name = "turkish-nlp-suite/BellaTurca"
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        configs = ["AkademikDerlem", "OzenliDerlem", "temiz-OSCAR", "temiz-mC4"]

    for cfg in configs:
        print(f"    BellaTurca config: {cfg}")
        ds = load_dataset(dataset_name, cfg, split="train", streaming=True)
        for row in ds:
            yield row
            if stop_requested:
                return


def stream_cosmos():
    ds = load_dataset("ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",
                      split="train", streaming=True)
    for row in ds:
        yield row
        if stop_requested:
            return


def stream_fineweb():
    ds = load_dataset("altaidevorg/fineweb-2-turkish-categorized",
                      split="train", streaming=True)
    for row in ds:
        yield {"text": row.get("text")}
        if stop_requested:
            return


# ===============================================================
# SOURCES
# ===============================================================

SOURCES = [
    ("BellaTurca", stream_bellaturca),
    ("Cosmos",     stream_cosmos),
    ("FineWeb2",   stream_fineweb),
]


# ===============================================================
# MAIN
# ===============================================================

def main():
    token = _read_hf_token()
    if not token:
        print("HuggingFace token not found. Run `huggingface-cli login` first.")
        sys.exit(1)

    api  = create_hf_repo(HF_REPO, private=PRIVATE_REPO)
    conn = init_sqlite(SQLITE_PATH)

    progress       = load_progress()
    resume_source  = progress.get("source")
    resume_shard   = progress.get("shard_id", 0)
    resume_records = progress.get("records_processed", 0)

    print(f"\nResume: source={resume_source}, shard_id={resume_shard}, records={resume_records:,}")
    print(f"Sources: {', '.join(n for n, _ in SOURCES)}\n")

    total_shards     = 0
    current_shard_id = resume_shard

    source_names = [n for n, _ in SOURCES]

    for name, gen_fn in SOURCES:

        # Skip sources that were already completed before the checkpoint
        if resume_source is not None and name != resume_source:
            if source_names.index(name) < source_names.index(resume_source):
                print(f"Skipping completed source: {name}")
                continue

        print(f"\n{'='*50}")
        print(f"Source: {name}")
        print(f"  Start shard_id : {current_shard_id}")
        if name == resume_source:
            print(f"  Records to skip: {resume_records:,}")
        print(f"{'='*50}")

        start_rec = resume_records if name == resume_source else 0

        produced = process_source(
            source_name    = name,
            dataset_iter   = gen_fn(),
            api            = api,
            conn           = conn,
            start_shard_id = current_shard_id,
            start_records  = start_rec,
        )

        print(f"  {name} -> {produced} shards uploaded.")
        current_shard_id += produced
        total_shards     += produced

        # Reset resume state after the checkpoint source is processed
        resume_source  = None
        resume_records = 0

        if stop_requested:
            print("Pipeline stopped. Checkpoint saved.")
            break

    if not stop_requested:
        meta = {
            "created_at"  : time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "hf_repo"     : HF_REPO,
            "shard_size"  : SHARD_SIZE,
            "sources"     : [n for n, _ in SOURCES],
            "total_shards": total_shards,
            "dedup"       : {
                "exact": "SHA-256 + SQLite",
                "near" : f"MinHash LSH (threshold={LSH_THRESHOLD}, perm={LSH_PERM})",
            },
        }
        upload_bytes_to_hf(
            api, HF_REPO, "metadata.json",
            io.BytesIO(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
        )
        clear_progress()
        print(f"\nDone. Total shards: {total_shards:,}")

    conn.close()


# ===============================================================
# ENTRY POINT
# ===============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "run_stream":
        print("Usage: python turkish_dataset_pipeline.py run_stream")
        sys.exit(1)
    main()