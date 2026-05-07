import json
import random
import re
import urllib.parse
from pathlib import Path
from typing import Dict, List, Set, Optional


SYLL = [
    "гар", "мор", "дран", "кхар", "зар", "гул", "раг", "тор", "дур", "зул",
    "кар", "кран", "брон", "шад", "тал", "кир", "нор", "док", "мак", "зир",
    "век", "тен", "рук", "грот", "вуль", "мок", "трак", "саг", "зок", "хел",
]

PLURAL_SUFFIXES = ["ары", "ены", "оны", "ксы", "доры", "римы", "роки", "дары", "тары"]


def _roll_core_word() -> str:
    n = random.choice([2, 3, 3, 4])
    parts = [random.choice(SYLL) for _ in range(n)]
    w = "".join(parts)

    if len(w) >= 6 and random.random() < 0.25:
        pos = random.choice([3, 4])
        w = w[:pos] + "'" + w[pos:]

    return w[:1].upper() + w[1:]


def _wow_like_person_or_place() -> str:
    if random.random() < 0.35:
        return _roll_core_word() + " " + _roll_core_word()
    return _roll_core_word()


def _wow_like_race_plural() -> str:
    base = _roll_core_word()
    base_low = base.lower()
    base_low = re.sub(r"[аеёиоуыэюя']{0,2}$", "", base_low)
    suffix = random.choice(PLURAL_SUFFIXES)
    out = base_low + suffix
    return out[:1].upper() + out[1:]


def _is_pluralish(src: str) -> bool:
    s = src.strip()
    return bool(re.search(r"(ены|оны|ары|иры|оры|ксы|роки|дары|тары|ы|и)$", s, flags=re.IGNORECASE))


def _title_from_url(url: str) -> str:
    if "/wiki/" not in url:
        return url.strip()

    tail = url.split("/wiki/", 1)[1]
    tail = tail.split("#", 1)[0].split("?", 1)[0]
    tail = urllib.parse.unquote(tail)
    tail = tail.replace("_", " ").strip()
    return tail


def _build_aliases(source_title: str) -> List[str]:
    t = re.sub(r"\s+", " ", source_title).strip()
    aliases = [t]

    tokens = t.split(" ")
    if len(tokens) >= 2:
        aliases.append(tokens[0])
        aliases.append(" ".join(tokens[:2]))
    else:
        aliases.append(t)

    uniq = []
    for a in aliases:
        a = a.strip()
        if len(a) >= 2 and a not in uniq:
            uniq.append(a)
    return uniq


def build_terms_map(raw_index_file: Path, terms_map_file: Path, seed: int = 7) -> None:
    random.seed(seed)

    idx = json.loads(raw_index_file.read_text(encoding="utf-8"))

    sources = []  
    for it in idx:
        if it.get("status") and it.get("status") != "ok":
            continue
        url = it["url"]
        sources.append(_title_from_url(url))

    extra = ["Азерот", "Орда", "Альянс"]
    sources.extend(extra)

    sources = sorted(set([s for s in sources if s]))

    used_targets = set()  
    mapping = []  

    for src in sources:
        pluralish = _is_pluralish(src)

        if pluralish:
            dst = _wow_like_race_plural()
        else:
            dst = _wow_like_person_or_place()

        while dst in used_targets:
            dst = _wow_like_race_plural() if pluralish else _wow_like_person_or_place()

        used_targets.add(dst)

        mapping.append(
            {
                "source": src,
                "target": dst,
                "aliases": _build_aliases(src),
                "replace_lowercase": pluralish,
            }
        )

    terms_map_file.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")