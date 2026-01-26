# scripts/utils.py
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    kb_dir: Path
    raw_dir: Path
    renamed_dir: Path
    wiki_urls_file: Path
    raw_index_file: Path
    terms_map_file: Path
    report_file: Path


def build_paths(project_root: Union[str, Path]) -> Paths:
    root = Path(project_root).resolve()
    data_dir = root / "data"
    kb_dir = root / "knowledge_base"
    raw_dir = kb_dir / "raw"
    renamed_dir = kb_dir / "renamed"

    return Paths(
        project_root=root,
        data_dir=data_dir,
        kb_dir=kb_dir,
        raw_dir=raw_dir,
        renamed_dir=renamed_dir,
        wiki_urls_file=data_dir / "wiki_urls.txt",
        raw_index_file=kb_dir / "raw_index.json",
        terms_map_file=kb_dir / "terms_map.json",
        report_file=kb_dir / "build_report.json",
    )


def ensure_dirs(paths: Paths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.kb_dir.mkdir(parents=True, exist_ok=True)


def wipe_artifacts(paths: Paths) -> None:
    # удаляем только артефакты пайплайна
    for d in [paths.raw_dir, paths.renamed_dir]:
        if d.exists():
            shutil.rmtree(d)

    for f in [paths.raw_index_file, paths.terms_map_file, paths.report_file]:
        if f.exists():
            f.unlink()