# scripts/apply_terms_map.py
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

RUS_LETTERS = r"А-Яа-яЁё"


def _compile_rules(term_map: List[Dict]) -> List[Dict]:
    rules = []

    for item in term_map:
        dst = item["target"]
        aliases = item.get("aliases") or [item["source"]]
        replace_lowercase = bool(item.get("replace_lowercase", False))

        for a in aliases:
            a = a.strip()
            if len(a) < 2:
                continue

            multiword = (" " in a)

            # Ловим: alias + русское окончание (0..10 символов)
            # Границы: не внутри слова
            pattern = r"(?<![{rus}])({alias})([{rus}]{{0,10}})(?![{rus}])".format(
                rus=RUS_LETTERS, alias=re.escape(a)
            )

            flags = re.UNICODE
            # если надо менять и lowercase, делаем re.IGNORECASE
            if replace_lowercase and not multiword:
                flags |= re.IGNORECASE

            rules.append(
                {
                    "alias": a,
                    "multiword": multiword,
                    "dst": dst,
                    "replace_lowercase": replace_lowercase,
                    "re": re.compile(pattern, flags=flags),
                }
            )

    # важнейшее: сначала длинные алиасы
    rules.sort(key=lambda r: len(r["alias"]), reverse=True)
    return rules


def _apply_rules(text: str, rules: List[Dict]) -> Tuple[str, Dict[str, int]]:
    stats = {}  # type: Dict[str, int]

    def repl_factory(rule: Dict):
        def _repl(m: re.Match) -> str:
            found = m.group(1)  # то, что совпало по alias (в реальном регистре)
            # suffix = m.group(2)  # окончание — игнорируем по ТЗ

            # Для много-словных имён (персонажи/локации) обычно хотим
            # заменять только “как имена собственные” (с заглавной),
            # чтобы не трогать случайные совпадения в тексте.
            if rule["multiword"] and found and not found[0].isupper():
                return m.group(0)

            # Для однословных:
            # - если replace_lowercase=True (расы/группы) — заменяем в любом регистре
            # - иначе: заменяем только с заглавной (как имя)
            if (not rule["multiword"]) and (not rule["replace_lowercase"]):
                if not (found and found[0].isupper()):
                    return m.group(0)

            key = rule["alias"]
            stats[key] = stats.get(key, 0) + 1

            # если совпадение было с маленькой буквы — делаем target тоже с маленькой
            dst = rule["dst"]
            if found and found[0].islower():
                dst = dst[:1].lower() + dst[1:]

            return dst

        return _repl

    out = text
    for rule in rules:
        out = rule["re"].sub(repl_factory(rule), out)
    return out, stats


def apply_terms_map(raw_dir: Path, renamed_dir: Path, terms_map_file: Path, report_file: Path) -> None:
    renamed_dir.mkdir(parents=True, exist_ok=True)

    term_map = json.loads(terms_map_file.read_text(encoding="utf-8"))
    rules = _compile_rules(term_map)

    report = {"files": [], "total_replacements": 0}

    for name in os.listdir(raw_dir):
        if not (name.endswith(".md") or name.endswith(".txt")):
            continue

        in_path = raw_dir / name
        out_path = renamed_dir / name

        txt = in_path.read_text(encoding="utf-8")
        new_txt, stats = _apply_rules(txt, rules)
        replaced = sum(stats.values())

        out_path.write_text(new_txt, encoding="utf-8")

        report["files"].append(
            {
                "file": name,
                "replacements": replaced,
                "top_aliases": sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10],
            }
        )
        report["total_replacements"] += replaced

    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")