import argparse
from pathlib import Path

from rag_core import make_bot_from_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Папка с faiss.index + chunks.jsonl")
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base", help="SentenceTransformer для эмбеддингов")
    ap.add_argument("--k", type=int, default=5, help="Top-k чанков")
    args = ap.parse_args()

    bot = make_bot_from_env(
        index_dir=Path(args.index_dir).resolve(),
        embed_model_name=args.embed_model,
        top_k=args.k,
    )

    print("RAG REPL. Напиши вопрос. Выход: Ctrl+C или пустая строка.\n")
    while True:
        q = input("> ").strip()
        if not q:
            break
        ans = bot.answer(q)
        print("\n" + ans + "\n")


if __name__ == "__main__":
    main()