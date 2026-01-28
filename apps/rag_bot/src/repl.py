import argparse
import os
from pathlib import Path

from rag_core import make_bot_from_env


def _env_default_str(name: str) -> str:
    return os.getenv(name, "").strip()


def _env_default_int(name: str, fallback: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return fallback
    try:
        return int(v)
    except ValueError:
        return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default=None, help="Папка с индексом (faiss.index + chunks.jsonl)")
    ap.add_argument("--embed_model", default=None, help="Модель эмбеддингов (SentenceTransformers)")
    ap.add_argument("--k", type=int, default=None, help="Top-K для поиска")
    args = ap.parse_args()

    index_dir = args.index_dir or _env_default_str("RAG_INDEX_DIR")
    embed_model = args.embed_model or _env_default_str("RAG_EMBED_MODEL") or "intfloat/multilingual-e5-base"
    top_k = args.k if args.k is not None else _env_default_int("RAG_TOP_K", 5)

    if not index_dir:
        raise SystemExit("index_dir is required (pass --index_dir or set RAG_INDEX_DIR)")

    bot = make_bot_from_env(index_dir=Path(index_dir), embed_model_name=embed_model, top_k=top_k)

    print("RAG REPL. Напиши вопрос. Выход: Ctrl+C или пустая строка.\n")
    while True:
        q = input("> ").strip()
        if not q:
            break
        ans = bot.answer(q)
        print("\n" + ans + "\n")


if __name__ == "__main__":
    main()