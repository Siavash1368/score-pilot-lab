"""Download the Lending Club dataset from Kaggle into data/raw/ and verify it.

Uses the kaggle CLI (token at ~/.kaggle/kaggle.json or ~/.kaggle/access_token):

    kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip

The Kaggle archive nests each CSV inside a same-named folder and also ships
.csv.gz copies, so after unzipping this script normalises the layout to
data/raw/<name>.csv, converts LCDataDictionary.xlsx to
data/reference/lc_data_dictionary.csv, and records file sizes and SHA-256
checksums in docs/decisions.md the first time it runs.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scorelab import config  # noqa: E402

ROOT = config.ROOT
RAW = config.RAW_DIR  # data/raw/ unless RAW_DATA_DIR is set in .env
REFERENCE = config.REFERENCE_DIR
DECISIONS = config.DOCS_DIR / "decisions.md"

DATASET = "wordsforthewise/lending-club"
CSV_NAMES = ["accepted_2007_to_2018Q4.csv", "rejected_2007_to_2018Q4.csv"]
DICTIONARY_CSV = REFERENCE / "lc_data_dictionary.csv"


def run_kaggle() -> None:
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW), "--unzip"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalise_layout() -> None:
    """Move each CSV to data/raw/<name>.csv, whatever nesting the archive used."""
    for name in CSV_NAMES:
        target = RAW / name
        if target.is_file():
            continue
        candidates = [p for p in RAW.rglob("*") if p.is_file() and p.name.lower() == name.lower()]
        if candidates:
            src = candidates[0]
            tmp = RAW / (name + ".tmp")
            shutil.move(str(src), tmp)
            if target.is_dir():
                shutil.rmtree(target)
            tmp.rename(target)
            continue
        gz = [p for p in RAW.rglob("*") if p.is_file() and p.name.lower() == name.lower() + ".gz"]
        if gz:
            print(f"Decompressing {gz[0].name}", flush=True)
            if target.is_dir():
                shutil.rmtree(target)
            with gzip.open(gz[0], "rb") as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    # Drop the duplicate .csv.gz copies once the plain CSVs are in place.
    for name in CSV_NAMES:
        gz = RAW / (name + ".gz")
        if (RAW / name).is_file() and gz.is_file():
            gz.unlink()


def xlsx_sheets_to_rows(path: Path) -> dict[str, list[list[str]]]:
    """Read every sheet of an .xlsx with the standard library (no openpyxl needed)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_target = {r.get("Id"): r.get("Target") for r in rels}
        sheets: dict[str, list[list[str]]] = {}
        for sheet in wb.find("m:sheets", ns):
            target = rel_target[sheet.get(f"{{{ns['r']}}}id")].lstrip("/")
            target = target if target.startswith("xl/") else "xl/" + target
            rows: list[list[str]] = []
            for row in ET.fromstring(z.read(target)).iter(f"{{{ns['m']}}}row"):
                values: dict[int, str] = {}
                for c in row.findall("m:c", ns):
                    col = 0
                    for ch in re.match(r"[A-Z]+", c.get("r")).group():
                        col = col * 26 + ord(ch) - 64
                    v = c.find("m:v", ns)
                    if c.get("t") == "s" and v is not None:
                        text = shared[int(v.text)]
                    elif c.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
                    else:
                        text = v.text if v is not None else ""
                    values[col - 1] = (text or "").strip()
                if values:
                    rows.append([values.get(i, "") for i in range(max(values) + 1)])
            sheets[sheet.get("name")] = rows
    return sheets


def save_dictionary() -> None:
    matches = [p for p in RAW.rglob("*") if p.is_file() and p.name.lower().startswith("lcdatadictionary")]
    if not matches:
        print("WARNING: LCDataDictionary.xlsx is not in the Kaggle archive; "
              "data/reference/lc_data_dictionary.csv was not written.", file=sys.stderr)
        return
    src = matches[0]
    REFERENCE.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".csv":
        shutil.copyfile(src, DICTIONARY_CSV)
    else:
        sheets = xlsx_sheets_to_rows(src)
        # The first sheet (LoanStats) describes the accepted-loan columns.
        name, rows = next(iter(sheets.items()))
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows if any(r)]
        rows = [r[:2] if width >= 2 and not any(x[2:] for x in rows) else r for r in rows]
        with open(DICTIONARY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"Saved sheet '{name}' ({len(rows) - 1} entries) to {DICTIONARY_CSV.relative_to(ROOT)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_checksums() -> None:
    text = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else "# Decisions log\n\n"
    if "Raw data checksums" in text:
        return
    lines = [f"## {dt.date.today().isoformat()} — Raw data checksums (first download)", "",
             f"Source: Kaggle `{DATASET}`.", "",
             "| File | Bytes | SHA-256 |", "|---|---:|---|"]
    for name in CSV_NAMES:
        p = RAW / name
        lines.append(f"| {name} | {p.stat().st_size:,} | `{sha256(p)}` |")
    DECISIONS.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("Recorded sizes and SHA-256 in docs/decisions.md")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    if not all((RAW / n).is_file() for n in CSV_NAMES):
        run_kaggle()
    normalise_layout()
    missing = [n for n in CSV_NAMES if not (RAW / n).is_file()]
    if missing:
        print(f"ERROR: missing after download: {missing}", file=sys.stderr)
        return 1
    for n in CSV_NAMES:
        print(f"OK  {RAW / n}  {(RAW / n).stat().st_size / 1e9:.2f} GB")
    save_dictionary()
    record_checksums()
    return 0


if __name__ == "__main__":
    sys.exit(main())
