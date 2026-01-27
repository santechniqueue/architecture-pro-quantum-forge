import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def e5_query(text: str) -> str:
    return "query: " + " ".join(text.split())


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="knowledge_base/index")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    index_dir = (repo_root / args.index_dir).resolve()

    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.jsonl"
    meta_path = index_dir / "index_meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = load_chunks(chunks_path)

    index = faiss.read_index(str(index_path))
    model = SentenceTransformer(args.model)

    q_emb = model.encode([e5_query(args.q)], normalize_embeddings=True, show_progress_bar=False)
    qv = np.asarray(q_emb, dtype=np.float32)

    scores, ids = index.search(qv, args.k)

    print(f"\nQuery: {args.q}")
    print(f"Top-{args.k} results:\n")

    for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
        item = chunks[idx]
        m = item["meta"]
        text = item["text"].replace("\n", " ").strip()
        preview = text[:300] + ("…" if len(text) > 300 else "")
        print(f"{rank}) score={score:.4f}")
        print(f"   source={m['source_path']} chunk_in_doc={m['chunk_in_doc']} pos=({m['start_char']},{m['end_char']})")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()