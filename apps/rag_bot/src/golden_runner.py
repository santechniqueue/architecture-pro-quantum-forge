import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag_core import make_bot_from_env, RagBot


def utc_ts_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Bad JSON on line {line_no} in {path}: {e}") from e
            items.append(obj)
    return items


def append_jsonl(path: Path, obj: Dict[str, Any], also_stdout: bool = True) -> None:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if also_stdout:
        print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def extract_answer_section(full: str) -> str:
    if not full:
        return ""
    s = full
    ia = s.find("\nОтвет:")
    isrc = s.find("\nИсточники:")
    if ia == -1:
        return s.strip()
    if isrc == -1:
        return s[ia + len("\nОтвет:") :].strip()
    return s[ia + len("\nОтвет:") : isrc].strip()


def has_required_sections(full: str) -> bool:
    if not full:
        return False
    s = full.strip()
    if not s.startswith("Ход мыслей:"):
        return False
    ia = s.find("\nОтвет:")
    isrc = s.find("\nИсточники:")
    return ia != -1 and isrc != -1 and ia < isrc


def normalize(s: str) -> str:
    return " ".join((s or "").lower().split())


def keyword_hits(text: str, keywords: Iterable[str]) -> Tuple[int, List[str]]:
    t = normalize(text)
    total = 0
    missing: List[str] = []
    for kw in keywords:
        k = normalize(kw)
        if not k:
            continue
        total += 1
        if k not in t:
            missing.append(kw)
    return total, missing


def any_forbidden(text: str, forbidden: Iterable[str]) -> List[str]:
    t = normalize(text)
    bad: List[str] = []
    for kw in forbidden:
        k = normalize(kw)
        if k and k in t:
            bad.append(kw)
    return bad


def sources_for_log(retrieved: List[Tuple[float, Any]], max_n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for score, ch in retrieved[: max_n if max_n > 0 else len(retrieved)]:
        meta = getattr(ch, "meta", {}) or {}
        out.append(
            {
                "source_path": meta.get("source_path", "unknown"),
                "chunk_in_doc": meta.get("chunk_in_doc", -1),
                "start_char": meta.get("start_char", -1),
                "end_char": meta.get("end_char", -1),
                "score": float(score),
            }
        )
    return out


def to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "да"):
            return True
        if s in ("0", "false", "no", "n", "нет"):
            return False
    return default


def get_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def run_one(
    bot: RagBot,
    case: Dict[str, Any],
    strict_no_answer: bool,
    log_max_sources: int,
) -> Dict[str, Any]:
    q = str(case.get("query", "")).strip()
    if not q:
        raise SystemExit(f"Case has empty query: {case}")

    expect_found = to_bool(case.get("expect_found", True), True)
    expected_keywords = get_list(case.get("expected_keywords"))
    forbidden_keywords = get_list(case.get("forbidden_keywords"))
    min_completeness = float(case.get("min_completeness", 0.0))
    case_id = str(case.get("id", "")) or q[:50]

    ts = utc_ts_iso()

    retrieved = bot.retrieve(q)
    chunks_found_count = len(retrieved)
    best_score: Optional[float] = retrieved[0][0] if retrieved else None
    sources = sources_for_log(retrieved, log_max_sources)

    err: Optional[str] = None
    ans: str = ""
    try:
        ans = bot.answer(q)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        ans = f"Ошибка при обработке запроса: {err}"

    answer_section = extract_answer_section(ans)
    answer_len = len(ans)

    format_ok = has_required_sections(ans)
    says_idk = "я не знаю" in normalize(answer_section)
    answer_found = bool(answer_section.strip()) and (not says_idk) and format_ok

    # полнота по ключевым словам
    total_kw, missing_kw = keyword_hits(answer_section, expected_keywords)
    if total_kw == 0:
        completeness = 1.0 if answer_section.strip() else 0.0
    else:
        completeness = (total_kw - len(missing_kw)) / float(total_kw)

    forbidden_hit = any_forbidden(answer_section, forbidden_keywords)

    if expect_found:
        passed = (
            answer_found
            and (best_score is None or best_score >= float(getattr(bot, "min_score", 0.0)))
            and completeness >= min_completeness
            and not forbidden_hit
        )
    else:
        if strict_no_answer:
            passed = says_idk or (best_score is not None and best_score < float(getattr(bot, "min_score", 0.0))) or chunks_found_count == 0
        else:
            passed = (not answer_found) or (best_score is not None and best_score < float(getattr(bot, "min_score", 0.0))) or chunks_found_count == 0

    out: Dict[str, Any] = {
        "timestamp": ts,
        "id": case_id,
        "query": q,
        "expect_found": expect_found,
        "chunks_found": chunks_found_count,
        "best_score": best_score,
        "answer_length": answer_len,
        "answer_found": answer_found,
        "completeness": round(float(completeness), 4),
        "min_completeness": min_completeness,
        "missing_keywords": missing_kw,
        "forbidden_hit": forbidden_hit,
        "pass": bool(passed),
        "sources": sources,
    }
    if err:
        out["error"] = err

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="Path to golden_set.jsonl")
    ap.add_argument("--out", default="logs/golden_results.jsonl", help="Output JSONL path")
    ap.add_argument("--index_dir", default=None, help="Папка с индексом (faiss.index + chunks.jsonl)")
    ap.add_argument("--embed_model", default=None, help="Модель эмбеддингов (SentenceTransformers)")
    ap.add_argument("--k", type=int, default=None, help="Top-K для поиска")
    ap.add_argument("--no_stdout", action="store_true", help="Не писать результаты в STDOUT")
    ap.add_argument("--strict_no_answer", type=int, default=1, help="1 = строго требовать 'Я не знаю' для expect_found=false")
    ap.add_argument("--log_max_sources", type=int, default=20, help="Сколько источников писать в результаты")
    args = ap.parse_args()

    golden_path = Path(args.golden)
    out_path = Path(args.out)

    if not golden_path.exists():
        raise SystemExit(f"Golden set not found: {golden_path}")

    index_dir = args.index_dir or os.getenv("RAG_INDEX_DIR", "").strip()
    embed_model = args.embed_model or os.getenv("RAG_EMBED_MODEL", "").strip() or "intfloat/multilingual-e5-base"
    top_k = args.k if args.k is not None else int(os.getenv("RAG_TOP_K", "5"))

    if not index_dir:
        raise SystemExit("index_dir is required (pass --index_dir or set RAG_INDEX_DIR)")

    bot = make_bot_from_env(index_dir=Path(index_dir), embed_model_name=embed_model, top_k=top_k)
    cases = read_jsonl(golden_path)

    total = 0
    passed = 0
    completeness_sum = 0.0

    for case in cases:
        total += 1
        r = run_one(
            bot=bot,
            case=case,
            strict_no_answer=bool(args.strict_no_answer),
            log_max_sources=int(args.log_max_sources),
        )
        if r.get("pass"):
            passed += 1
        completeness_sum += float(r.get("completeness", 0.0))

        append_jsonl(out_path, r, also_stdout=(not args.no_stdout))

    avg_completeness = (completeness_sum / float(total)) if total else 0.0
    summary = {
        "timestamp": utc_ts_iso(),
        "type": "summary",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / float(total), 4) if total else 0.0,
        "avg_completeness": round(avg_completeness, 4),
        "golden": str(golden_path),
        "out": str(out_path),
    }
    append_jsonl(out_path, summary, also_stdout=True)


if __name__ == "__main__":
    main()
