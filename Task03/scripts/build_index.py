import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


def extract_title(md: str, fallback: str) -> str:
    for raw in md.splitlines():
        line = raw.lstrip()
        if not line.startswith("#"):
            continue
        i = 0
        n = len(line)
        while i < n and line[i] == "#":
            i += 1
        if i == 0:
            continue
        if i < n and line[i] == " ":
            title = line[i:].strip()
            if title:
                return title
    return fallback


def md_to_text(md: str) -> str:
    """
    Markdown -> плоский текст, без regex:
    - вырезает fenced code blocks ```...``` и ~~~...~~~
    - вырезает inline code `...`
    - убирает HTML-теги <...>
    - ![alt](url) -> alt
    - [text](url) -> text
    - убирает маркеры заголовков/цитат/списков (best-effort)
    - нормализует пробелы/пустые строки
    """
    s = md

    s = _strip_fenced_code(s)
    s = _strip_inline_code(s)
    s = _strip_html_tags(s)
    s = _rewrite_links_and_images(s)
    s = _strip_md_line_prefixes(s)
    s = _normalize_whitespace(s)

    return s.strip()


def _strip_fenced_code(s: str) -> str:
    """
    Удаляет fenced blocks на ``` или ~~~.
    Условия открытия/закрытия:
    - последовательность из 3+ одинаковых символов (` или ~) в начале строки
    """
    out: List[str] = []
    i = 0
    n = len(s)

    in_fence = False
    fence_char = ""
    fence_len = 0

    while i < n:
        is_line_start = (i == 0) or (s[i - 1] == "\n")

        if is_line_start and i < n and (s[i] == "`" or s[i] == "~"):
            ch = s[i]
            j = i
            while j < n and s[j] == ch:
                j += 1
            run = j - i

            if run >= 3:
                if not in_fence:
                    in_fence = True
                    fence_char = ch
                    fence_len = run
                else:
                    if ch == fence_char and run >= fence_len:
                        in_fence = False
                        fence_char = ""
                        fence_len = 0

                while j < n and s[j] != "\n":
                    j += 1
                i = j
                continue

        if not in_fence:
            out.append(s[i])
        i += 1

    return "".join(out)


def _strip_inline_code(s: str) -> str:
    out: List[str] = []
    i = 0
    n = len(s)
    in_code = False

    while i < n:
        ch = s[i]
        if ch == "`":
            in_code = not in_code
            i += 1
            continue
        if not in_code:
            out.append(ch)
        i += 1

    return "".join(out)


def _strip_html_tags(s: str) -> str:
    out: List[str] = []
    i = 0
    n = len(s)
    in_tag = False

    while i < n:
        ch = s[i]
        if ch == "<":
            in_tag = True
            i += 1
            continue
        if ch == ">":
            if in_tag:
                in_tag = False
                i += 1
                continue
        if not in_tag:
            out.append(ch)
        i += 1

    return "".join(out)


def _rewrite_links_and_images(s: str) -> str:
    """
    Best-effort замена:
      - ![alt](url) -> alt
      - [text](url) -> text
    Без regex, с поддержкой экранирования \"\\\".
    """
    out: List[str] = []
    i = 0
    n = len(s)

    while i < n:
        
        if s[i] == "!" and i + 1 < n and s[i + 1] == "[":
            alt, after_br = _parse_bracketed(s, i + 1)
            if alt is not None and after_br < n and s[after_br] == "(":
                _, after_par = _parse_parenthesized(s, after_br)
                if after_par is not None:
                    alt_txt = alt.strip()
                    if alt_txt:
                        out.append(alt_txt)
                    i = after_par
                    continue

        if s[i] == "[":
            txt, after_br = _parse_bracketed(s, i)
            if txt is not None and after_br < n and s[after_br] == "(":
                _, after_par = _parse_parenthesized(s, after_br)
                if after_par is not None:
                    t = txt.strip()
                    if t:
                        out.append(t)
                    i = after_par
                    continue

        out.append(s[i])
        i += 1

    return "".join(out)


def _parse_bracketed(s: str, start_i: int) -> Tuple[str | None, int]:
    """
    start_i указывает на '['.
    Возвращает (content, index_after_closing_bracket) или (None, start_i).
    """
    if start_i >= len(s) or s[start_i] != "[":
        return None, start_i

    i = start_i + 1
    buf: List[str] = []
    depth = 1
    n = len(s)

    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            buf.append(s[i + 1])
            i += 2
            continue
        if ch == "[":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return "".join(buf), i + 1
            buf.append(ch)
            i += 1
            continue

        buf.append(ch)
        i += 1

    return None, start_i


def _parse_parenthesized(s: str, start_i: int) -> Tuple[str | None, int | None]:
    """
    start_i указывает на '('.
    Возвращает (content, index_after_closing_paren) или (None, None).
    """
    if start_i >= len(s) or s[start_i] != "(":
        return None, None

    i = start_i + 1
    buf: List[str] = []
    depth = 1
    n = len(s)

    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            buf.append(s[i + 1])
            i += 2
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return "".join(buf), i + 1
            buf.append(ch)
            i += 1
            continue

        buf.append(ch)
        i += 1

    return None, None


def _strip_md_line_prefixes(s: str) -> str:
    lines = s.splitlines()
    out_lines: List[str] = []

    for raw in lines:
        line = raw.lstrip()

        if line.startswith("#"):
            i = 0
            n = len(line)
            while i < n and line[i] == "#":
                i += 1
            if i < n and line[i] == " ":
                line = line[i + 1 :].lstrip()

        if line.startswith(">"):
            i = 0
            n = len(line)
            while i < n and line[i] == ">":
                i += 1
            if i < n and line[i] == " ":
                line = line[i + 1 :].lstrip()
            else:
                line = line[i:].lstrip()

        if line.startswith("- ") or line.startswith("* ") or line.startswith("+ "):
            line = line[2:].lstrip()

        j = 0
        n = len(line)
        while j < n and line[j].isdigit():
            j += 1
        if j > 0 and j + 1 < n and line[j] == "." and line[j + 1] == " ":
            line = line[j + 2 :].lstrip()

        out_lines.append(line)

    return "\n".join(out_lines)


def _normalize_whitespace(s: str) -> str:
    s = s.replace("\t", " ")

    out_chars: List[str] = []
    prev_space = False
    for ch in s:
        if ch == " ":
            if not prev_space:
                out_chars.append(ch)
            prev_space = True
        else:
            out_chars.append(ch)
            prev_space = False
    s = "".join(out_chars)

    lines = [ln.strip() for ln in s.splitlines()]
    out_lines: List[str] = []
    empty_streak = 0
    for ln in lines:
        if ln == "":
            empty_streak += 1
            if empty_streak <= 1:
                out_lines.append("")
        else:
            empty_streak = 0
            out_lines.append(ln)

    return "\n".join(out_lines)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def iter_md_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.md"):
        if p.is_file():
            yield p


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )


def chunk_with_offsets(text: str, splitter: RecursiveCharacterTextSplitter) -> List[Tuple[str, int, int]]:
    """
    Возвращает список (chunk_text, start_char, end_char).
    Оффсеты считаем сами через find().
    """
    chunks = splitter.split_text(text)
    res: List[Tuple[str, int, int]] = []
    cursor = 0

    for ch in chunks:
        if not ch.strip():
            continue

        idx = text.find(ch, cursor)
        if idx == -1:
            start = -1
            end = -1
        else:
            start = idx
            end = idx + len(ch)
            cursor = end

        res.append((ch, start, end))

    return res


def e5_passage(text: str) -> str:
    return "passage: " + " ".join(text.split())


def e5_query(text: str) -> str:
    return "query: " + " ".join(text.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_dir", default="Task02/knowledge_base/renamed", help="Папка с KB .md")
    ap.add_argument("--out_dir", default="Task03/index", help="Куда сохранять индекс и метаданные")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="Модель эмбеддингов")
    ap.add_argument("--chunk_size", type=int, default=1400, help="Размер чанка в символах (RecursiveCharacterTextSplitter)")
    ap.add_argument("--chunk_overlap", type=int, default=200, help="Оверлап чанков в символах")
    ap.add_argument("--batch", type=int, default=64, help="Batch size для encode")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = (repo_root / args.kb_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "faiss.index"
    chunks_path = out_dir / "chunks.jsonl"
    meta_path = out_dir / "index_meta.json"

    splitter = build_splitter(args.chunk_size, args.chunk_overlap)

    files = sorted(iter_md_files(kb_dir))
    if not files:
        raise SystemExit(f"No .md files found in: {kb_dir}")

    chunk_records: List[ChunkRecord] = []
    for fp in tqdm(files, desc="Chunking files"):
        rel = fp.relative_to(kb_dir).as_posix()
        raw_md = read_text(fp)
        raw_txt = md_to_text(raw_md)
        if not raw_txt.strip():
            continue

        pieces = chunk_with_offsets(raw_txt, splitter)
        title = extract_title(raw_md, fp.stem)

        for i, (ch, start, end) in enumerate(pieces):
            clean = ch.strip()
            if not clean:
                continue

            cid = _sha1(f"{rel}::{i}::{clean[:200]}")
            chunk_records.append(
                ChunkRecord(
                    chunk_id=cid,
                    text=clean,
                    meta={
                        "source_path": rel,
                        "chunk_in_doc": i,
                        "start_char": start,
                        "end_char": end,
                        "title": title,
                    },
                )
            )

    if not chunk_records:
        raise SystemExit("No chunks produced (all files empty?)")

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model)

    passages = [e5_passage(r.text) for r in chunk_records]

    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(passages), args.batch), desc="Embedding"):
        batch_texts = passages[i : i + args.batch]
        em = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(np.asarray(em, dtype=np.float32))
    X = np.vstack(embeddings)

    dt = time.perf_counter() - t0

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(index_path))

    with chunks_path.open("w", encoding="utf-8") as f:
        for r in chunk_records:
            f.write(json.dumps({"chunk_id": r.chunk_id, "text": r.text, "meta": r.meta}, ensure_ascii=False) + "\n")

    meta = {
        "model": args.model,
        "embedding_dim": dim,
        "kb_dir": str(kb_dir),
        "files": len(files),
        "chunks": len(chunk_records),
        "chunk_size_chars": args.chunk_size,
        "chunk_overlap_chars": args.chunk_overlap,
        "index_type": "faiss.IndexFlatIP",
        "build_seconds": round(dt, 3),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDONE")
    print(f"Index:   {index_path}")
    print(f"Chunks:  {chunks_path}")
    print(f"Meta:    {meta_path}")
    print(f"Chunks:  {len(chunk_records)} | Dim: {dim} | Build time: {meta['build_seconds']}s")


if __name__ == "__main__":
    main()