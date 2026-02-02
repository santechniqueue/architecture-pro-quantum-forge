import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from rag_core import make_bot_from_env, RagBot

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Я RAG-бот. Спроси меня о базе знаний.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Просто отправь вопрос сообщением. Я отвечу с источниками.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot: RagBot = context.bot_data["rag_bot"]
    q = (update.message.text or "").strip()
    if not q:
        return
    try:
        ans = bot.answer(q)
    except Exception as e:
        ans = f"Ошибка при обработке запроса: {type(e).__name__}: {e}"
    await update.message.reply_text(ans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default=None, help="Папка с индексом (faiss.index + chunks.jsonl)")
    ap.add_argument("--embed_model", default=None, help="Модель эмбеддингов (SentenceTransformers)")
    ap.add_argument("--k", type=int, default=None, help="Top-K для поиска")
    args = ap.parse_args()

    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN").strip()
    if not token:
        print("TELEGRAM_BOT_TOKEN is empty. Put it in .env or export env var.")
        sys.exit(1)

    index_dir = args.index_dir or os.getenv("RAG_INDEX_DIR").strip()
    embed_model = args.embed_model or os.getenv("RAG_EMBED_MODEL").strip() or "intfloat/multilingual-e5-base"
    top_k = args.k if args.k is not None else int(os.getenv("RAG_TOP_K", 5))

    if not index_dir:
        raise SystemExit("index_dir is required (pass --index_dir or set RAG_INDEX_DIR)")

    rag_bot = make_bot_from_env(index_dir=Path(index_dir), embed_model_name=embed_model, top_k=top_k)

    app = ApplicationBuilder().token(token).build()
    app.bot_data["rag_bot"] = rag_bot

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Telegram bot started.")
    app.run_polling()


if __name__ == "__main__":
    main()