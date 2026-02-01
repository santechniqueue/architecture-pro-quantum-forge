import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stable_faiss_id(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    u64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(u64 & 0x7FFFFFFFFFFFFFFF)


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def is_e5_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "e5" in name


def e5_query(text: str) -> str:
    return "query: " + normalize_ws(text)


def e5_passage(text: str) -> str:
    return "passage: " + normalize_ws(text)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_md_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.md"):
        if p.is_file():
            yield p


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_title(md: str, fallback: str) -> str:
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            s2 = s.lstrip("#").strip()
            return s2 if s2 else fallback
    return fallback


def _strip_md_inline(s: str) -> str:
    if not s:
        return s

    out_chars: List[str] = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if ch == "<":
            j = s.find(">", i + 1)
            if j != -1:
                i = j + 1
                continue

        if ch == "`":
            j = s.find("`", i + 1)
            if j != -1:
                out_chars.append(" ")
                out_chars.append(s[i + 1 : j])
                out_chars.append(" ")
                i = j + 1
                continue

        if ch == "!" and (i + 1) < n and s[i + 1] == "[":
            end_alt = s.find("]", i + 2)
            if end_alt != -1 and (end_alt + 1) < n and s[end_alt + 1] == "(":
                end_url = s.find(")", end_alt + 2)
                if end_url != -1:
                    alt = s[i + 2 : end_alt]
                    out_chars.append(" ")
                    out_chars.append(alt)
                    out_chars.append(" ")
                    i = end_url + 1
                    continue

        if ch == "[":
            end_txt = s.find("]", i + 1)
            if end_txt != -1 and (end_txt + 1) < n and s[end_txt + 1] == "(":
                end_url = s.find(")", end_txt + 2)
                if end_url != -1:
                    txt = s[i + 1 : end_txt]
                    out_chars.append(" ")
                    out_chars.append(txt)
                    out_chars.append(" ")
                    i = end_url + 1
                    continue

        out_chars.append(ch)
        i += 1

    return "".join(out_chars)


def md_to_text(md: str) -> str:
    lines_out: List[str] = []

    in_fence = False
    fence = "```"

    for line in md.splitlines():
        s = line.rstrip("\n")

        if s.strip().startswith(fence):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        s = s.strip()

        if not s:
            lines_out.append("")
            continue

        while s.startswith("#"):
            s = s[1:]
        s = s.lstrip()

        if s.startswith(">"):
            s = s[1:].lstrip()

        for bullet in ("- ", "* ", "+ "):
            if s.startswith(bullet):
                s = s[len(bullet):]
                break

        if s and s[0].isdigit():
            dot_pos = s.find(". ")
            if dot_pos in (1, 2):
                s = s[dot_pos + 2 :]

        s = _strip_md_inline(s)
        lines_out.append(s)

    txt = "\n".join(lines_out)
    while "\n\n\n" in txt:
        txt = txt.replace("\n\n\n", "\n\n")

    cleaned_lines: List[str] = []
    for ln in txt.splitlines():
        cleaned_lines.append(" ".join(ln.split()))
    txt2 = "\n".join(cleaned_lines)

    parts = [p.strip() for p in txt2.split("\n\n") if p.strip()]
    return "\n\n".join(parts).strip()


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # in characters
        chunk_overlap=chunk_overlap,    # in characters
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )


def chunk_with_offsets(text: str, splitter: RecursiveCharacterTextSplitter) -> List[Tuple[str, int, int]]:
    chunks = splitter.split_text(text)
    res: List[Tuple[str, int, int]] = []
    cursor = 0
    for ch in chunks:
        if not ch.strip():
            continue
        idx = text.find(ch, cursor)
        if idx == -1:
            res.append((ch, -1, -1))
            continue
        start = idx
        end = idx + len(ch)
        cursor = end
        res.append((ch, start, end))
    return res


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    faiss_id: int
    text: str
    meta: Dict[str, Any]


@dataclass
class IndexState:
    model: str
    embed_dim: int
    index_kind: str          # flat | ivf
    nlist: int
    nprobe: int
    files: Dict[str, Dict[str, Any]]  # rel_path -> {sha1, chunk_ids, faiss_ids}

    @staticmethod
    def empty(model: str, embed_dim: int, index_kind: str, nlist: int, nprobe: int) -> "IndexState":
        return IndexState(
            model=model,
            embed_dim=embed_dim,
            index_kind=index_kind,
            nlist=nlist,
            nprobe=nprobe,
            files={},
        )

    @staticmethod
    def load(path: Path) -> Optional["IndexState"]:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return IndexState(
            model=obj.get("model", ""),
            embed_dim=int(obj.get("embed_dim", 0)),
            index_kind=obj.get("index_kind", "flat"),
            nlist=int(obj.get("nlist", 0)),
            nprobe=int(obj.get("nprobe", 0)),
            files=dict(obj.get("files", {})),
        )

    def save(self, path: Path) -> None:
        obj = {
            "model": self.model,
            "embed_dim": self.embed_dim,
            "index_kind": self.index_kind,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "files": self.files,
        }
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _effective_nlist(requested_nlist: int, train_points: int) -> int:
    if train_points <= 0:
        return max(1, requested_nlist)
    max_reasonable = max(1, train_points // 39)
    return max(1, min(int(requested_nlist), int(max_reasonable)))


def build_faiss_index(index_kind: str, dim: int, nlist: int) -> faiss.Index:
    kind = (index_kind or "flat").lower().strip()
    if kind == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIDMap2(ivf)
    flat = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(flat)


def _unwrap_ivf(index: faiss.Index) -> Optional[faiss.IndexIVF]:
    try:
        ivf = faiss.extract_index_ivf(index)
        return ivf
    except Exception:
        return None


def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    ivf = _unwrap_ivf(index)
    if ivf is not None:
        ivf.nprobe = int(nprobe)


def ensure_trained_ivf(index: faiss.Index, train_vectors: np.ndarray) -> None:
    ivf = _unwrap_ivf(index)
    if ivf is None:
        return

    if ivf.is_trained:
        return

    if train_vectors is None or train_vectors.size == 0:
        raise RuntimeError("IVF index is not trained and there is no data to train on.")

    X_train = np.ascontiguousarray(train_vectors, dtype=np.float32)

    if X_train.shape[0] < max(2, int(ivf.nlist)):
        raise RuntimeError(
            f"Not enough training points for IVF: have {X_train.shape[0]}, need at least nlist={ivf.nlist}."
        )

    ivf.train(X_train)

    if not ivf.is_trained:
        raise RuntimeError("IVF training did not complete (is_trained is still False).")


def remove_ids(index: faiss.Index, ids: List[int]) -> int:
    if not ids:
        return 0
    arr = np.asarray(ids, dtype=np.int64)
    selector = faiss.IDSelectorArray(arr.size, faiss.swig_ptr(arr))
    return int(index.remove_ids(selector))


def embed_passages(
    model: SentenceTransformer,
    passages: List[str],
    batch_size: int,
) -> np.ndarray:
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(passages), batch_size), desc="Embedding", leave=False):
        bt = passages[i : i + batch_size]
        e = model.encode(bt, normalize_embeddings=True, show_progress_bar=False)
        embs.append(np.asarray(e, dtype=np.float32))
    if not embs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(embs)


def build_chunk_records(
    kb_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> Tuple[List[ChunkRecord], Dict[str, Dict[str, Any]]]:
    records: List[ChunkRecord] = []
    manifest: Dict[str, Dict[str, Any]] = {}

    files = sorted(iter_md_files(kb_dir))
    for fp in tqdm(files, desc="Chunking", leave=False):
        rel = fp.relative_to(kb_dir).as_posix()
        fhash = file_sha1(fp)
        raw_md = read_text(fp)
        title = extract_title(raw_md, fp.stem)
        plain = md_to_text(raw_md)
        if not plain.strip():
            manifest[rel] = {"sha1": fhash, "chunk_ids": [], "faiss_ids": [], "title": title}
            continue

        pieces = chunk_with_offsets(plain, splitter)
        chunk_ids: List[str] = []
        faiss_ids: List[int] = []

        for i, (ch, start, end) in enumerate(pieces):
            clean = ch.strip()
            if not clean:
                continue

            chunk_key = f"{rel}::{i}::{clean[:200]}"
            cid = sha1_hex(chunk_key)
            fid = stable_faiss_id(chunk_key)

            chunk_ids.append(cid)
            faiss_ids.append(fid)

            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    faiss_id=fid,
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

        manifest[rel] = {"sha1": fhash, "chunk_ids": chunk_ids, "faiss_ids": faiss_ids, "title": title}

    return records, manifest


def load_chunks_jsonl(path: Path) -> List[ChunkRecord]:
    recs: List[ChunkRecord] = []
    if not path.exists():
        return recs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta", {})
            fid = obj.get("faiss_id")
            if fid is None:
                rel = meta.get("source_path", "unknown")
                i = meta.get("chunk_in_doc", 0)
                key = f"{rel}::{i}::{(obj.get('text') or '')[:200]}"
                fid = stable_faiss_id(key)
            recs.append(
                ChunkRecord(
                    chunk_id=obj["chunk_id"],
                    faiss_id=int(fid),
                    text=obj.get("text", ""),
                    meta=meta,
                )
            )
    return recs


def write_chunks_jsonl(path: Path, records: List[ChunkRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {"chunk_id": r.chunk_id, "faiss_id": r.faiss_id, "text": r.text, "meta": r.meta},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_dir", default="knowledge_base/renamed", help="Папка с KB .md")
    ap.add_argument("--out_dir", default="knowledge_base/index", help="Куда сохранять индекс и метаданные")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="Модель эмбеддингов (SentenceTransformers)")
    ap.add_argument("--chunk_size", type=int, default=1400, help="Размер чанка в символах (RecursiveCharacterTextSplitter)")
    ap.add_argument("--chunk_overlap", type=int, default=200, help="Оверлап чанков в символах")
    ap.add_argument("--batch", type=int, default=64, help="Batch size для encode")

    ap.add_argument("--index_kind", choices=["flat", "ivf"], default="ivf", help="Тип FAISS индекса")
    ap.add_argument("--mode", choices=["full", "incremental"], default="full", help="full: пересобрать полностью; incremental: обновить изменённые")
    ap.add_argument("--nlist", type=int, default=256, help="(IVF) число кластеров")
    ap.add_argument("--nprobe", type=int, default=8, help="(IVF) число просматриваемых кластеров при поиске")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = (repo_root / args.kb_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "faiss.index"
    chunks_path = out_dir / "chunks.jsonl"
    meta_path = out_dir / "index_meta.json"
    state_path = out_dir / "index_state.json"

    if not kb_dir.exists():
        raise SystemExit(f"KB dir does not exist: {kb_dir}")

    splitter = build_splitter(args.chunk_size, args.chunk_overlap)

    t0 = time.perf_counter()
    new_records_all, new_manifest = build_chunk_records(kb_dir, splitter)

    if not new_records_all:
        raise SystemExit("No chunks produced (KB empty after cleaning?)")

    embed_model = SentenceTransformer(args.model)

    if is_e5_model(args.model):
        passages = [e5_passage(r.text) for r in new_records_all]
    else:
        passages = [normalize_ws(r.text) for r in new_records_all]

    X = embed_passages(embed_model, passages, args.batch)
    if X.size == 0:
        raise SystemExit("Embeddings are empty. Something went wrong.")

    dim = int(X.shape[1])

    mode = args.mode.lower().strip()
    kind = args.index_kind.lower().strip()

    if mode == "full" or not index_path.exists() or not chunks_path.exists():
        eff_nlist = _effective_nlist(args.nlist, int(X.shape[0])) if kind == "ivf" else 0

        index = build_faiss_index(kind, dim, eff_nlist if kind == "ivf" else 0)
        set_nprobe(index, args.nprobe)

        if kind == "ivf":
            ensure_trained_ivf(index, X)

        ids = np.asarray([r.faiss_id for r in new_records_all], dtype=np.int64)
        index.add_with_ids(X, ids)

        faiss.write_index(index, str(index_path))
        write_chunks_jsonl(chunks_path, new_records_all)

        dt = time.perf_counter() - t0

        state = IndexState.empty(
            model=args.model,
            embed_dim=dim,
            index_kind=kind,
            nlist=eff_nlist if kind == "ivf" else 0,
            nprobe=int(args.nprobe),
        )
        state.files = new_manifest
        state.save(state_path)

        meta = {
            "mode": "full",
            "model": args.model,
            "embedding_dim": dim,
            "kb_dir": str(kb_dir),
            "files": len(new_manifest),
            "chunks": len(new_records_all),
            "chunk_size_chars": args.chunk_size,
            "chunk_overlap_chars": args.chunk_overlap,
            "index_kind": kind,
            "index_type": type(index).__name__,
            "nlist": eff_nlist if kind == "ivf" else None,
            "nprobe": int(args.nprobe) if kind == "ivf" else None,
            "build_seconds": round(dt, 3),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print("DONE (full)")
        print(f"Index:  {index_path}")
        print(f"Chunks: {chunks_path}")
        print(f"State:  {state_path}")
        print(f"Meta:   {meta_path}")
        print(f"Chunks: {len(new_records_all)} | Dim: {dim} | Time: {meta['build_seconds']}s")
        if kind == "ivf":
            print(f"IVF: nlist={eff_nlist} (requested {args.nlist}) | nprobe={args.nprobe}")
        return

    old_state = IndexState.load(state_path)
    if old_state is None:
        print("No index_state.json found -> fallback to full rebuild.")
        args.mode = "full"
        return main()

    if old_state.model != args.model:
        print(f"Embedding model changed: {old_state.model} -> {args.model}. Rebuild required.")
        args.mode = "full"
        return main()

    index = faiss.read_index(str(index_path))
    set_nprobe(index, args.nprobe)

    old_records = load_chunks_jsonl(chunks_path)

    old_files = set(old_state.files.keys())
    new_files = set(new_manifest.keys())

    deleted_files = sorted(old_files - new_files)
    added_files = sorted(new_files - old_files)
    common_files = sorted(old_files & new_files)

    changed_files: List[str] = []
    for rel in common_files:
        if old_state.files.get(rel, {}).get("sha1") != new_manifest.get(rel, {}).get("sha1"):
            changed_files.append(rel)

    remove_id_list: List[int] = []
    for rel in deleted_files + changed_files:
        remove_id_list.extend(old_state.files.get(rel, {}).get("faiss_ids", []) or [])
    removed = remove_ids(index, remove_id_list)

    keep_files = set(common_files) - set(changed_files)
    kept_records: List[ChunkRecord] = [r for r in old_records if r.meta.get("source_path") in keep_files]

    add_or_replace_files = set(added_files) | set(changed_files)
    new_records_subset: List[ChunkRecord] = [r for r in new_records_all if r.meta.get("source_path") in add_or_replace_files]

    if new_records_subset:
        if is_e5_model(args.model):
            passages2 = [e5_passage(r.text) for r in new_records_subset]
        else:
            passages2 = [normalize_ws(r.text) for r in new_records_subset]
        X2 = embed_passages(embed_model, passages2, args.batch)

        ensure_trained_ivf(index, X2)

        ids2 = np.asarray([r.faiss_id for r in new_records_subset], dtype=np.int64)
        index.add_with_ids(X2, ids2)

    final_records = kept_records + new_records_subset
    final_records.sort(key=lambda r: (r.meta.get("source_path", ""), int(r.meta.get("chunk_in_doc", 0))))

    faiss.write_index(index, str(index_path))
    write_chunks_jsonl(chunks_path, final_records)

    for rel in deleted_files:
        old_state.files.pop(rel, None)
    for rel in added_files + changed_files:
        old_state.files[rel] = new_manifest[rel]

    old_state.nprobe = int(args.nprobe)
    old_state.save(state_path)

    dt = time.perf_counter() - t0

    meta = {
        "mode": "incremental",
        "model": args.model,
        "embedding_dim": dim,
        "kb_dir": str(kb_dir),
        "files_total": len(old_state.files),
        "chunks_total": len(final_records),
        "files_added": len(added_files),
        "files_changed": len(changed_files),
        "files_deleted": len(deleted_files),
        "vectors_removed": removed,
        "vectors_added": len(new_records_subset),
        "chunk_size_chars": args.chunk_size,
        "chunk_overlap_chars": args.chunk_overlap,
        "index_kind": kind,
        "index_type": type(index).__name__,
        "nprobe": int(args.nprobe) if _unwrap_ivf(index) is not None else None,
        "update_seconds": round(dt, 3),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE (incremental)")
    print(f"Index:  {index_path}")
    print(f"Chunks: {chunks_path}")
    print(f"State:  {state_path}")
    print(f"Meta:   {meta_path}")
    print(
        f"Files: +{len(added_files)} ~{len(changed_files)} -{len(deleted_files)} | "
        f"Vectors: -{removed} +{len(new_records_subset)} | Time: {meta['update_seconds']}s"
    )


if __name__ == "__main__":
    main()