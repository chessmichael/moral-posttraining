"""Moral Foundations Vignettes (Clifford et al., 2015): real human norming data, used for evaluation.

Table 1 of the paper gives 132 short "You see someone..." scenarios with the % of respondents
who classified each under each foundation and the mean wrongness rating (0-4). The vignettes
use the original MFT foundations, so only some map cleanly onto MFQ-2:

  Care (e/p,h/p,a) -> Care, Loyalty -> Loyalty, Authority -> Authority, Sanctity -> Purity
  Fairness -> mostly cheating, i.e. closest to Proportionality (kept separate as "Fairness")
  Liberty, Social Norms -> no MFQ-2 counterpart (Social Norms is the non-moral baseline)

    python -m mft.vignettes parse   # data/raw/clifford2015_pmc.html -> data/raw/vignettes.csv
"""
from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
from pathlib import Path

SRC = Path("data/raw/clifford2015_pmc.html")
OUT = Path("data/raw/vignettes.csv")
CLASS_COLUMNS = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty", "Not Wrong"]
TO_MFQ2 = {"Care": "Care", "Loyalty": "Loyalty", "Authority": "Authority", "Sanctity": "Purity",
           "Fairness": "Fairness", "Liberty": "Liberty", "Social Norms": "Social Norms"}


class _TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse(html: str) -> list[dict]:
    parser = _TableRows()
    parser.feed(html)
    table = next(t for t in parser.tables if any(r[:1] == ["Foundation"] for r in t))
    rows = []
    for r in table:
        if len(r) != 10 or not r[9].replace(".", "").isdigit():
            continue
        text, foundation, *pcts, wrong = r
        base = foundation.split(" (")[0]
        rows.append({
            "id": f"mfv-{len(rows):03d}", "text": text, "foundation": foundation, "mfq2": TO_MFQ2[base],
            **{f"pct_{c.replace(' ', '_').lower()}": float(p.rstrip(" %")) / 100 for c, p in zip(CLASS_COLUMNS, pcts)},
            "wrong": float(wrong),
        })
    return rows


def load(path: Path = OUT) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["parse"])
    args = parser.parse_args()
    rows = parse(SRC.read_text(encoding="utf-8"))
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["foundation"]] = counts.get(r["foundation"], 0) + 1
    print(f"wrote {len(rows)} vignettes to {OUT}: {counts}")


if __name__ == "__main__":
    main()
