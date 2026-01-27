import argparse
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from env import EnvConfig
from rag_core import make_bot_from_env


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я RAG-бот по базе знаний.\n"
        "Просто задай вопрос, и я отвечу на основе найденных фрагментов.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Папка с faiss.index + chunks.jsonl")
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base", help="SentenceTransformer для эмбеддингов")
    ap.add_argument("--k", type=int, default=5, help="Top-k чанков")
    ap.add_argument("--env_path", default=None, help="Путь к .env (опционально)")
    args = ap.parse_args()

    cfg = EnvConfig.from_env(env_path=args.env_path)

    bot = make_bot_from_env(
        index_dir=Path(args.index_dir).resolve(),
        embed_model_name=args.embed_model,
        top_k=args.k,
    )

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        q = update.message.text.strip()
        if not q:
            return
        ans = bot.answer(q)
        await update.message.reply_text(ans)

    app = Application.builder().token(cfg.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram bot started.")
    app.run_polling()


if __name__ == "__main__":
    main()