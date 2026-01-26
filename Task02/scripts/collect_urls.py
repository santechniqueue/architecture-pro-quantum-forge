import json
import random
import requests
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

BASE = "https://wowwiki.fandom.com"   # <-- ВАЖНО: без /ru
API = BASE + "/ru/api.php"           # <-- ВАЖНО: /ru/api.php

CATEGORIES = [
    "Категория:Персонажи",
    "Категория:Артефакты",
    "Категория:События",
]


def _api_get(params: Dict[str, str]) -> Dict:
    url = API + "?" + urllib.parse.urlencode(params)
    r = requests.get(url, headers={"User-Agent": "kb-builder/1.0"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _page_url(title: str) -> str:
    t = title.strip()

    if t.startswith("http://") or t.startswith("https://"):
        return t

    if t.startswith("/"):
        t = t[1:]

    # если уже ru/wiki/... — не добавляем второй раз
    if t.lower().startswith("ru/wiki/"):
        path = t[len("ru/wiki/"):]
    else:
        path = t

    path = path.replace(" ", "_")
    return BASE + "/ru/wiki/" + urllib.parse.quote(path)


def _collect_from_category(cat_title: str, limit: int = 500) -> List[str]:
    out = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmlimit": "500",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = _api_get(params)
        for it in data.get("query", {}).get("categorymembers", []):
            title = it.get("title", "")
            if not title or ":" in title:
                continue
            out.append(title)
            if len(out) >= limit:
                return out

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            return out


def collect_urls(urls_file: Path, target_count: int = 40, seed: int = 42) -> None:
    random.seed(seed)

    titles = []
    for cat in CATEGORIES:
        titles.extend(_collect_from_category(cat, limit=500))

    titles = sorted(set(titles))
    random.shuffle(titles)
    titles = titles[:target_count]

    urls = [_page_url(t) for t in titles]
    urls_file.parent.mkdir(parents=True, exist_ok=True)
    urls_file.write_text("\n".join(urls) + "\n", encoding="utf-8")