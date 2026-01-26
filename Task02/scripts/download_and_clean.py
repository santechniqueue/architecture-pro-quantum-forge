# scripts/download_and_clean.py
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import requests
from bs4 import BeautifulSoup
from slugify import slugify


UA = "kb-builder/1.0"

STOP_HEADINGS = {
    "Галерея",
    "Видео",
    "Заметки",
    "Рекомендации",
    "Внешние ссылки",
    "Ссылки",
    "Примечания",
    "Источники",
    "См. также",
}

def _make_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({"User-Agent": UA})
    return session


def _download_html(session: requests.Session, url: str, timeout: tuple) -> str:
    # stream=True + чтение по кускам снижает шанс IncompleteRead на больших/длинных ответах
    with session.get(url, timeout=timeout, stream=True) as r:
        if r.status_code == 404:
            raise requests.HTTPError("404", response=r)
        r.raise_for_status()

        chunks = []
        for chunk in r.iter_content(chunk_size=64 * 1024, decode_unicode=False):
            if chunk:
                chunks.append(chunk)

        raw = b"".join(chunks)
        enc = r.encoding or "utf-8"
        return raw.decode(enc, errors="replace")



def _stable_name(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    base = slugify(url.split("/wiki/")[-1])[:60] or "entity"
    return base + "-" + h + ".md"


def _heading_text(tag) -> str:
    """
    Достаём чистый текст заголовка секции (обычно в span.mw-headline),
    чтобы корректно матчить "Видео", "Галерея", "Заметки" и т.п.
    """
    span = tag.select_one(".mw-headline")
    if span:
        t = span.get_text(" ", strip=True)
    else:
        t = tag.get_text(" ", strip=True)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def _cut_after_stop_heading(content: BeautifulSoup) -> None:
    stop_lower = {s.lower() for s in STOP_HEADINGS}

    for hdr in content.find_all(["h2", "h3", "h4"]):
        title = _heading_text(hdr).lower()
        # иногда заголовки бывают типа "Заметки и факты" — режем по startswith
        if title in stop_lower or any(title.startswith(s + " ") for s in stop_lower):
            # удаляем все элементы, начиная с этого заголовка
            cur = hdr
            while cur is not None:
                nxt = cur.next_sibling
                try:
                    cur.extract()
                except Exception:
                    pass
                cur = nxt
            break



def _clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one(".mw-parser-output") or soup.body or soup

    # 1) Убираем “шапки”/переадресации/подсказки вида:
    #    "Для просмотра ... перейдите ..."
    for sel in [
        ".hatnote", ".dablink", ".rellink", ".notice", ".noprint",
        ".mw-editsection", ".printfooter",
    ]:
        for t in content.select(sel):
            t.decompose()

    # 2) Убираем инфобоксы/TOC/навигацию/таблицы/референсы
    for sel in [
        "table", "aside", "nav", "header", "footer",
        ".portable-infobox", ".pi-item", ".toc",
        "sup.reference", ".reference", ".navbox",
    ]:
        for t in content.select(sel):
            t.decompose()

    # 3) Убираем медиа и их подписи (в т.ч. то, что у тебя вылезает как
    #    "Khadgar with A'dal in the movie")
    for sel in [
        "figure", "figcaption",
        "iframe", "video", "audio",
        ".gallery", ".wikia-gallery", ".gallerybox", ".gallerytext",
        ".thumb", ".thumbcaption",
        ".fandom-video", ".video", ".mw-video-player",
    ]:
        for t in content.select(sel):
            t.decompose()

    # 4) Отрезаем хвосты по секциям (Видео/Галерея/Заметки/Внешние ссылки/…)
    _cut_after_stop_heading(content)

    # 5) ВАЖНО: собираем только “основной” текст — абзацы и списки
    blocks = []
    for child in content.find_all(["p", "ul", "ol"], recursive=True):
        txt = child.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue

        # отсекаем очень короткий “мусор” (часто остатки от ссылок/подписей)
        if len(txt) < 3:
            continue

        blocks.append(txt)

    text = "\n\n".join(blocks)

    # 6) Финальная нормализация: убираем повторные пробелы и пустые блоки
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_urls(wiki_urls_file: Path) -> List[str]:
    urls = []
    for line in wiki_urls_file.read_text(encoding="utf-8").splitlines():
        u = line.strip()
        if not u or u.startswith("#"):
            continue
        urls.append(u)
    return urls


def download_and_clean(
    project_root: Path,
    wiki_urls_file: Path,
    raw_dir: Path,
    raw_index_file: Path,
    polite_delay_s: float = 0.5,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = _read_urls(wiki_urls_file)
    index = []  # type: List[Dict[str, str]]

    session = _make_session()

    # (connect_timeout, read_timeout)
    timeout = (10, 45)

    for url in urls:
        ok = False
        last_error = None

        # локальные "ручные" повторы поверх retry (на случай редких ошибок чтения)
        for attempt in range(1, 4):
            try:
                html = _download_html(session, url, timeout=timeout)
                text = _clean_text_from_html(html)

                fname = _stable_name(url)
                abs_path = raw_dir / fname
                abs_path.write_text(text + "\n", encoding="utf-8")

                rel_path = abs_path.relative_to(project_root).as_posix()
                index.append({"url": url, "file": rel_path, "status": "ok"})
                ok = True
                break

            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as e:
                last_error = e
                # небольшой бэкофф
                time.sleep(1.2 * attempt)
                continue

            except requests.HTTPError as e:
                # 404/прочее — не ретраим бесконечно
                last_error = e
                break

            except Exception as e:
                last_error = e
                break

        if not ok:
            # не валим весь пайплайн из-за одной страницы
            index.append({"url": url, "file": "", "status": "failed", "error": str(last_error)})
            print("SKIP:", url, "-", last_error)

        time.sleep(polite_delay_s)

    raw_index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")