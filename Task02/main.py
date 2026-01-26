# main.py
from pathlib import Path

from scripts.utils import build_paths, ensure_dirs, wipe_artifacts
from scripts.download_and_clean import download_and_clean
from scripts.build_terms_map import build_terms_map
from scripts.apply_terms_map import apply_terms_map
from scripts.verify_uniqueness import verify_uniqueness

def run(paths, recollect: bool = False) -> None:
    if recollect:
        ensure_dirs(paths)
        wipe_artifacts(paths)

        # 1) скачиваем и очищаем из заранее заполненного data/wiki_urls.txt
        download_and_clean(
            project_root=paths.project_root,
            wiki_urls_file=paths.wiki_urls_file,
            raw_dir=paths.raw_dir,
            raw_index_file=paths.raw_index_file,
            polite_delay_s=0.5,
        )

        # 2) строим terms_map (raw_index.json содержит относительные пути)
        build_terms_map(
            raw_index_file=paths.raw_index_file,
            terms_map_file=paths.terms_map_file,
            seed=7,
        )

    # 3) применяем подмену + отчёт
    apply_terms_map(
        raw_dir=paths.raw_dir,
        renamed_dir=paths.renamed_dir,
        terms_map_file=paths.terms_map_file,
        report_file=paths.report_file,
    )

    # 4) проверяем утечки
    verify_uniqueness(
        renamed_dir=paths.renamed_dir,
        terms_map_file=paths.terms_map_file,
    )

    print("✅ DONE")
    print("- renamed:", paths.renamed_dir)
    print("- terms_map:", paths.terms_map_file)
    print("- report:", paths.report_file)
    print("- raw_index:", paths.raw_index_file)


if __name__ == "__main__":
    run(build_paths(Path(__file__).parent), recollect=False)