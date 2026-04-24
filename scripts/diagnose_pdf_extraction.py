#!/usr/bin/env python3
"""diagnose_pdf_extraction.py — inspect raw MuPDF block extraction for a PDF.

Run this against the McNichol 2025 GRUMP paper to find why "5000" and
"GTGYCAGCMGCCGCGGTAA" are not appearing in the extracted text.

Usage:
    python scripts/diagnose_pdf_extraction.py <path-to-pdf>

Example:
    python scripts/diagnose_pdf_extraction.py \
      notrack/reference_bank/GRUMP/10.1038s41597_025_05423_9_1494.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGETS = ["5000", "GTGYCAGCMGCCGCGGTAA", "CCGYCAATTYMTTTRAGTTT", "sequencing depth"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_pdf_extraction.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    import fitz

    doc = fitz.open(str(pdf_path))
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(doc)}\n")

    # -----------------------------------------------------------------------
    # Pass 1: raw get_text() — does the target appear at all?
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("PASS 1 — raw get_text() (default mode)")
    print("=" * 60)
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()

    for t in TARGETS:
        idx = raw_text.lower().find(t.lower())
        if idx >= 0:
            snippet = raw_text[max(0, idx - 80):idx + 120].replace("\n", " ")
            print(f"  FOUND '{t}' at char {idx}:")
            print(f"    ...{snippet}...")
        else:
            print(f"  NOT FOUND: '{t}'")

    # -----------------------------------------------------------------------
    # Pass 2: block extraction — which page and block contains target?
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PASS 2 — get_text('blocks') — locate by page and position")
    print("=" * 60)

    found_pages = {t: [] for t in TARGETS}

    for page_num, page in enumerate(doc, start=1):
        page_h = page.rect.height
        page_w = page.rect.width
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))

        for b in blocks:
            if b[6] != 0:
                continue
            text = b[4]
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            for t in TARGETS:
                if t.lower() in text.lower():
                    found_pages[t].append({
                        "page": page_num,
                        "x0": round(x0), "y0": round(y0),
                        "x1": round(x1), "y1": round(y1),
                        "page_h": round(page_h),
                        "page_w": round(page_w),
                        "y_frac": round(y0 / page_h, 3),
                        "text_snippet": text[:200].replace("\n", " "),
                    })

    for t in TARGETS:
        hits = found_pages[t]
        if hits:
            print(f"\n  '{t}' found in {len(hits)} block(s):")
            for h in hits:
                print(f"    page={h['page']}  y={h['y0']}/{h['page_h']} "
                      f"(y_frac={h['y_frac']})  "
                      f"x={h['x0']}–{h['x1']}")
                print(f"    text: {h['text_snippet']!r}")
        else:
            print(f"\n  '{t}' NOT FOUND in any block")

    # -----------------------------------------------------------------------
    # Pass 3: show the current _extract_text_from_file output and check
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PASS 3 — current _extract_text_from_file() output")
    print("=" * 60)

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    try:
        from cmap_agent.sync.kb_sync import _extract_text_from_file
        extracted = _extract_text_from_file(pdf_path)
        if extracted:
            print(f"  Extracted {len(extracted)} chars total")
            for t in TARGETS:
                idx = extracted.lower().find(t.lower())
                if idx >= 0:
                    snippet = extracted[max(0, idx-80):idx+120].replace("\n", " ")
                    print(f"  FOUND '{t}' at char {idx}:")
                    print(f"    ...{snippet}...")
                else:
                    print(f"  NOT FOUND in extracted text: '{t}'")
        else:
            print("  Extraction returned None/empty")
    except Exception as exc:
        print(f"  Error running extractor: {exc}")

    # -----------------------------------------------------------------------
    # Pass 4: show all block y-positions on the page where target appears
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PASS 4 — all blocks on the page(s) containing target terms")
    print("         (helps identify if footer filter is clipping content)")
    print("=" * 60)

    target_pages = set()
    for t in TARGETS:
        for h in found_pages[t]:
            target_pages.add(h["page"])

    import re as _re
    HEADER_PAT = _re.compile(
        r"(scientific\s+data|nature\s+communications|nature\s+methods"
        r"|plos\s+one|frontiers\s+in\s+\w|molecular\s+ecology\s+resources"
        r"|doi\.org/10\.\d{4}|www\.nature\.com|www\.frontiersin\.org)",
        _re.I,
    )

    for page_num in sorted(target_pages)[:3]:  # limit to first 3 pages
        page = doc[page_num - 1]
        page_h = page.rect.height
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        print(f"\n  Page {page_num} (height={round(page_h)}):")
        for b in blocks:
            if b[6] != 0:
                continue
            text = b[4].strip()
            if not text:
                continue
            x0, y0 = b[0], b[1]
            y_frac = y0 / page_h
            in_header = y_frac < 0.10
            in_footer = y_frac > 0.92
            is_meta = bool(HEADER_PAT.search(text))
            dropped = (in_header or in_footer) and is_meta
            has_target = any(t.lower() in text.lower() for t in TARGETS)
            mark = " <-- TARGET" if has_target else ""
            drop_mark = " [DROPPED]" if dropped else ""
            print(f"    y={round(y0):4d}/{round(page_h)} ({y_frac:.2f})  "
                  f"x={round(x0):4d}  {text[:60]!r}{mark}{drop_mark}")

    doc.close()


if __name__ == "__main__":
    main()
