import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def e5_query(text: str) -> str:
    return "query: " + normalize_ws(text)


def e5_passage(text: str) -> str:
    return "passage: " + normalize_ws(text)

def _clamp_text(s: str, max_chars: int) -> str:
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"

def _fewshot_answer(self, ch) -> str:
    return _clamp_text(ch.text, 240)


def _build_context_blocks(hits, max_total_chars: int, max_chunk_chars: int):
    blocks = []
    total = 0

    for score, rec in hits:
        text = rec.get("text") if isinstance(rec, dict) else getattr(rec, "text", "")
        meta = rec.get("meta") if isinstance(rec, dict) else getattr(rec, "meta", {})

        if not text:
            continue

        text = _clamp_text(text, max_chunk_chars)

        header = f"Источник: {meta.get('source_path', 'unknown')} | chunk={meta.get('chunk_in_doc', '?')} | score={score:.4f}"
        block = header + "\n" + text

        add = len(block) + 2
        if total + add > max_total_chars:
            break

        blocks.append(block)
        total += add

    return "\n\n---\n\n".join(blocks), len(blocks), total


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


class FaissIndex:
    def __init__(self, index: faiss.Index, chunks: List[Chunk]) -> None:
        self.index = index
        self.chunks = chunks

    @staticmethod
    def load(index_dir: Path) -> "FaissIndex":
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

        index = faiss.read_index(str(index_path))

        chunks: List[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=obj["chunk_id"],
                        text=obj["text"],
                        meta=obj.get("meta", {}),
                    )
                )

        return FaissIndex(index=index, chunks=chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[float, Chunk]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, idxs = self.index.search(q, k)
        res: List[Tuple[float, Chunk]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.chunks):
                continue
            res.append((float(score), self.chunks[idx]))
        return res


@dataclass
class RagBot:
    index: FaissIndex
    embedder: SentenceTransformer
    llm: OpenAI
    llm_model: str

    top_k: int = 5
    min_score: float = 0.35

    def embed_query(self, q: str) -> np.ndarray:
        # normalize_embeddings=True => cosine ~ inner product
        vec = self.embedder.encode([e5_query(q)], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec[0], dtype=np.float32)

    def retrieve(self, q: str) -> List[Tuple[float, Chunk]]:
        qv = self.embed_query(q)
        return self.index.search(qv, self.top_k)

    def build_fewshot_examples(self) -> List[Dict[str, str]]:
        example_questions = [
            "Кто такая Норкирзок Зулзирдранкхар?",
            "Что такое Бро'нвекмор Макморхел?",
        ]
        shots: List[Dict[str, str]] = []

        for q in example_questions:
            hits = self.retrieve(q)
            if not hits:
                continue
            score, ch = hits[0]
            answer = self._format_answer_from_chunk(q, score, ch)
            shots.append({"q": q, "a": answer})

        return shots[:2]

    def _format_answer_from_chunk(self, q: str, score: float, ch: Chunk) -> str:
        src = ch.meta.get("source_path", "unknown")
        pos = f"{ch.meta.get('start_char', -1)}..{ch.meta.get('end_char', -1)}"
        return (
            f"Я нашёл релевантный фрагмент в источнике {src} (позиция {pos}). "
            f"По нему: {ch.text[:350].rstrip()}..."
        )

    def build_prompt(self, user_q: str, retrieved: List[Tuple[float, Chunk]]) -> List[Dict[str, str]]:
        system = (
            "Ты RAG-помощник. Отвечай ТОЛЬКО на основе контекста ниже.\n"
            "Если в контексте нет ответа — скажи: «Я не знаю».\n"
            "\n"
            "ФОРМАТ ВЫВОДА ОБЯЗАТЕЛЕН:\n"
            "Ход мыслей:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "(допустимо 4 шага)\n"
            "Ответ: ...\n"
            "Источники:\n"
            "- <source_path> | chunk=<chunk_in_doc> | pos=<start..end>\n"
            "\n"
            "Правила:\n"
            "- Всегда выводи 3–4 шага, даже если ответ короткий.\n"
            "- Шаги должны ссылаться на факты из контекста (что нашёл и почему это отвечает).\n"
            "- Не повторяй инструкции промпта.\n"
        )

        msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

        shots = self.build_fewshot_examples()
        for s in shots:
            msgs.append({"role": "user", "content": f"Q: {s['q']}"})
            msgs.append({"role": "assistant", "content": f"A: {s['a']}"})

        if retrieved:
            ctx_lines: List[str] = []
            for i, (score, ch) in enumerate(retrieved, start=1):
                src = ch.meta.get("source_path", "unknown")
                chunk_no = ch.meta.get("chunk_in_doc", -1)
                pos = f"{ch.meta.get('start_char', -1)}..{ch.meta.get('end_char', -1)}"
                ctx_lines.append(
                    f"[{i}] score={score:.4f} source={src} chunk_in_doc={chunk_no} pos={pos}\n{ch.text}"
                )
            context_block = "\n\n".join(ctx_lines)
        else:
            context_block = ""

        user = (
            f"Вопрос пользователя: {user_q}\n\n"
            f"Контекст из базы знаний:\n{context_block}\n\n"
            "Ответь по требованиям."
        )
        msgs.append({"role": "user", "content": user})
        return msgs

    def answer(self, user_q: str) -> str:
        retrieved = self.retrieve(user_q)

        if not retrieved:
            return "Я не знаю.\n\nИсточники: (ничего не найдено)"

        best_score = retrieved[0][0]
        if best_score < self.min_score:
            return "Я не знаю.\n\nИсточники: (релевантных фрагментов недостаточно)"

        messages = self.build_prompt(user_q, retrieved)

        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content


def make_bot_from_env(
    index_dir: Path,
    embed_model_name: str,
    top_k: int,
) -> RagBot:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")

    base_url = base_url.replace("http://localhost", "http://127.0.0.1")
    base_url = base_url.replace("https://localhost", "https://127.0.0.1")

    if base_url.endswith(":11434"):
        base_url = base_url + "/v1"


    if not api_key:
        if base_url.startswith("http://localhost:11434") or base_url.startswith("http://127.0.0.1:11434"):
            api_key = "ollama"
        else:
            raise SystemExit("OPENAI_API_KEY is empty. Set env var OPENAI_API_KEY.")

    index = FaissIndex.load(index_dir=index_dir)

    embedder = SentenceTransformer(embed_model_name)
    llm = OpenAI(api_key=api_key, base_url=base_url)

    return RagBot(
        index=index,
        embedder=embedder,
        llm=llm,
        llm_model=model_name,
        top_k=top_k,
    )