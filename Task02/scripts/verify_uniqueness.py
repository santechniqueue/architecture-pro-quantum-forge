import json
import os
import re
from pathlib import Path
from typing import List

RUS = r"А-Яа-яЁё"


def verify_uniqueness(renamed_dir: Path, terms_map_file: Path) -> None:
    term_map = json.loads(terms_map_file.read_text(encoding="utf-8"))
    sources = [t["source"] for t in term_map if t.get("source")]  
    sources.sort(key=len, reverse=True)

    parts = []
    for fn in os.listdir(renamed_dir):
        p = renamed_dir / fn
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
    blob = "\n".join(parts)

    leaks = []
    for s in sources:
        pat = re.compile(r"(?<![{rus}]){term}([{rus}]{{0,8}})(?![{rus}])".format(
            rus=RUS, term=re.escape(s)
        ))
        if pat.search(blob):
            leaks.append(s)

    if leaks:
        preview = "\n".join("- " + x for x in leaks[:50])
        raise RuntimeError("LEAKS FOUND: original terms still present:\n" + preview)