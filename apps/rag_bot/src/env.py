import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class EnvConfig:
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str

    @staticmethod
    def from_env(env_path: Optional[str] = None) -> "EnvConfig":
        load_dotenv(dotenv_path=env_path)

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()

        if not telegram_bot_token:
            raise SystemExit("TELEGRAM_BOT_TOKEN is empty. Set env var TELEGRAM_BOT_TOKEN.")

        return EnvConfig(
            telegram_bot_token=telegram_bot_token,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        )