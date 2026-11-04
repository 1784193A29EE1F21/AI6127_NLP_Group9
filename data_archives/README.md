# Source corpus and human annotations

The four `wiki_output_*.zip` archives contain all original WikiExtractor files in the matching `wiki_output/AA`, `AB`, `AC` and `AD` folders. The system's original corpus build defaults to `AA/wiki_00` through `AA/wiki_99`; the other folders are included to preserve the complete source directory.

`two_human_annotations.zip` contains the original 200 annotation JSON files (100 for each annotator) and the integrated two-human evaluation dataset `rag/data/eval/human_pair/eval_qa.jsonl`. It does not contain AI annotations.

From the repository root, extract all archives while preserving their internal paths:

```bash
python -c "from pathlib import Path; from zipfile import ZipFile; [ZipFile(p).extractall('.') for p in sorted(Path('data_archives').glob('*.zip'))]"
```

This creates `wiki_output/`, `6127-data/6127-data/`, and `rag/data/eval/human_pair/`. Extraction can overwrite files at those paths; back up existing files first. The source files and evaluation data are ignored by Git after extraction.

`manifest.json` records file counts, archive sizes and SHA-256 checksums. Corpus and evaluation data may require independent agreement review; the archives do not establish an inter-annotator agreement rate.
