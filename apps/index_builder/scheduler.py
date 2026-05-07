import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import logging
import json

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("index_scheduler")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False

    return logger


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KiB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.1f} MiB"
    gb = mb / 1024.0
    return f"{gb:.2f} GiB"


def _run_build_index_stream(logger: logging.Logger, cmd: list[str]) -> tuple[int, str]:
    stderr_lines: list[str] = []

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert p.stdout is not None
    assert p.stderr is not None

    for line in p.stdout:
        line = line.rstrip("\n")
        if line:
            logger.info(line)

    for line in p.stderr:
        line = line.rstrip("\n")
        if line:
            stderr_lines.append(line)
            logger.warning(line)

    rc = p.wait()
    return rc, "\n".join(stderr_lines)


def _log_update_summary(logger: logging.Logger, index_dir: Path) -> None:
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "index_meta.json"

    meta = _read_json_safe(meta_path)

    new_chunks = meta.get("vectors_added", None)
    chunks_total = meta.get("chunks_total", None)
    update_seconds = meta.get("update_seconds", None)

    files_added = meta.get("files_added", None)
    files_changed = meta.get("files_changed", None)
    files_deleted = meta.get("files_deleted", None)

    idx_size = None
    if index_path.exists():
        idx_size = index_path.stat().st_size

    logger.info("SUMMARY:")
    if new_chunks is not None and chunks_total is not None:
        logger.info(f"  new_chunks={new_chunks} | chunks_total={chunks_total}")
    elif chunks_total is not None:
        logger.info(f"  chunks_total={chunks_total}")
    else:
        logger.info("  chunks_total=<unknown> (index_meta.json not readable)")

    if idx_size is not None:
        logger.info(f"  index_size={_human_size(idx_size)} ({idx_size} bytes)")
    else:
        logger.info("  index_size=<missing faiss.index>")

    if files_added is not None or files_changed is not None or files_deleted is not None:
        logger.info(f"  files: +{files_added} ~{files_changed} -{files_deleted}")

    if update_seconds is not None:
        logger.info(f"  update_seconds={update_seconds}")


def run_update_with_retries(logger: logging.Logger) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "apps" / "index_builder" / "build_index.py"

    kb_dir = _env("KB_DIR", "knowledge_base/renamed")
    out_dir = _env("INDEX_DIR", "knowledge_base/index")
    model = _env("EMBED_MODEL", "intfloat/multilingual-e5-base")

    chunk_size = _env("CHUNK_SIZE", "1400")
    chunk_overlap = _env("CHUNK_OVERLAP", "200")
    batch = _env("BATCH", "64")

    index_kind = _env("INDEX_KIND", "ivf")
    nlist = _env("NLIST", "64")
    nprobe = _env("NPROBE", "8")

    retry_delay = int(_env("RETRY_DELAY_SECONDS", "60"))
    max_attempts = int(_env("RETRY_MAX_ATTEMPTS", "3"))

    index_dir = (repo_root / out_dir).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--kb_dir", kb_dir,
        "--out_dir", out_dir,
        "--model", model,
        "--chunk_size", chunk_size,
        "--chunk_overlap", chunk_overlap,
        "--batch", batch,
        "--mode", "incremental",
        "--index_kind", index_kind,
        "--nlist", nlist,
        "--nprobe", nprobe,
    ]

    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
        start = _utc_ts()
        logger.info(f"[{start}] START update attempt={attempt}/{max_attempts}")
        logger.info("CMD: " + " ".join(cmd))

        try:
            rc, stderr_text = _run_build_index_stream(logger, cmd)
            end = _utc_ts()
            logger.info(f"[{end}] END update rc={rc}")

            if stderr_text:
                logger.warning("STDERR was not empty (see lines above).")

            if rc != 0:
                raise RuntimeError(f"build_index.py failed with rc={rc}")

            _log_update_summary(logger, index_dir)
            return

        except Exception as e:
            last_err = repr(e)
            logger.exception(f"Attempt {attempt} failed: {last_err}")

            if attempt < max_attempts:
                logger.info(f"Retry in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All attempts failed. Last error: {last_err}")
                raise


def main() -> None:
    load_dotenv()

    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "index_update.log"

    logger = setup_logging(log_path)

    mode = str(_env("SCHEDULE_MODE", "demo"))

    if mode == "demo":
        run_update_with_retries(logger)
        return

    hour = int(_env("SCHEDULE_HOUR", "6"))
    minute = int(_env("SCHEDULE_MINUTE", "0"))

    sched = BlockingScheduler()
    sched.add_job(lambda: run_update_with_retries(logger), "cron", hour=hour, minute=minute)

    logger.info(f"Scheduler started (cron {hour:02d}:{minute:02d}).")
    sched.start()


if __name__ == "__main__":
    main()