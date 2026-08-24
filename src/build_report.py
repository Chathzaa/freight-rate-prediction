"""Render report/report.md to a PDF.

    python -m src.build_report

Supports the subset of Markdown the report uses: headings, paragraphs, bullet
lists, pipe tables, images and inline bold/code.  reportlab is the only extra
dependency and it renders locally, so nothing leaves the machine.
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "report.md"
TARGET = ROOT / "report" / "freight_rate_report.pdf"

INK = colors.HexColor("#111111")
INK_2 = colors.HexColor("#4a4a4a")
RULE = colors.HexColor("#d5d5d0")
HEAD_BG = colors.HexColor("#eef2f3")
ACCENT = colors.HexColor("#064A56")

PAGE_W, PAGE_H = LETTER
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


def styles() -> dict:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "body": ParagraphStyle("body", parent=base, fontName="Helvetica", fontSize=9.6,
                               leading=14.2, textColor=INK, spaceAfter=7, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=19,
                             leading=23, textColor=ACCENT, spaceBefore=0, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=13,
                             leading=17, textColor=ACCENT, spaceBefore=15, spaceAfter=7),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=10.5,
                             leading=14, textColor=INK, spaceBefore=10, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", parent=base, fontName="Helvetica", fontSize=9.6,
                                 leading=14.2, textColor=INK, leftIndent=14,
                                 bulletIndent=3, spaceAfter=4),
        "cell": ParagraphStyle("cell", parent=base, fontName="Helvetica", fontSize=8.4,
                               leading=11.4, textColor=INK, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base, fontName="Helvetica-Bold",
                                   fontSize=8.4, leading=11.4, textColor=INK, spaceAfter=0),
        "caption": ParagraphStyle("caption", parent=base, fontName="Helvetica-Oblique",
                                  fontSize=8.2, leading=11, textColor=INK_2, spaceAfter=10),
    }


def inline(text: str) -> str:
    """Markdown emphasis to reportlab markup, with the raw text escaped first."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.8">\1</font>', text)
    return text


def starts_block(line: str) -> bool:
    """True only for real block openers.

    Checked with the exact prefixes rather than a character set, so a paragraph
    line that merely begins with a minus sign (a negative number, say) is not
    mistaken for a bullet and dropped.
    """
    s = line.strip()
    return s.startswith(("#", "|", "![", "- "))


def parse(markdown: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            blocks.append(("image", (match.group(1), match.group(2))))
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append((f"h{min(level, 3)}", stripped.lstrip("#").strip()))
            i += 1
            continue

        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") and c for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].strip().startswith("- ") or
                                      (lines[i].startswith("  ") and lines[i].strip() and items)):
                if lines[i].strip().startswith("- "):
                    items.append(lines[i].strip()[2:])
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append(("bullets", items))
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(paragraph)))
    return blocks


def build_table(rows: list[list[str]], st: dict) -> Table:
    header, *body = rows
    ncols = len(header)

    # Width each column by its longest cell, then scale to the text width.
    weights = []
    for c in range(ncols):
        longest = max((stringWidth(re.sub(r"[*`]", "", r[c]), "Helvetica", 8.4)
                       for r in rows if c < len(r)), default=40)
        weights.append(max(38.0, min(longest + 14, 250.0)))
    scale = CONTENT_W / sum(weights)
    widths = [w * scale for w in weights]

    data = [[Paragraph(inline(c), st["cellhead"]) for c in header]]
    data += [[Paragraph(inline(c), st["cell"]) for c in row] for row in body]

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_image(alt: str, relative: str, st: dict) -> list:
    path = (SOURCE.parent / relative).resolve()
    if not path.is_file():
        return [Paragraph(f"[missing image: {relative}]", st["caption"])]
    from PIL import Image as PILImage  # pillow ships with matplotlib installs
    with PILImage.open(path) as im:
        w, h = im.size
    width = CONTENT_W
    height = width * h / w
    max_h = 3.5 * inch
    if height > max_h:
        height, width = max_h, max_h * w / h
    flow = [Spacer(1, 4), Image(str(path), width=width, height=height, hAlign="LEFT")]
    if alt:
        flow.append(Spacer(1, 3))
        flow.append(Paragraph(alt, st["caption"]))
    return flow


def decorate(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(INK_2)
    canvas.drawString(MARGIN, 0.5 * inch, "Freight Rate Prediction, ML Engineer assessment")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * inch, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 0.66 * inch, PAGE_W - MARGIN, 0.66 * inch)
    canvas.restoreState()


def main() -> None:
    st = styles()
    story = []
    for kind, payload in parse(SOURCE.read_text(encoding="utf-8")):
        if kind == "image":
            story += build_image(*payload, st)
        elif kind == "table":
            story += [Spacer(1, 3), build_table(payload, st), Spacer(1, 10)]
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(inline(item), st["bullet"], bulletText="•"))
            story.append(Spacer(1, 5))
        elif kind in ("h1", "h2", "h3"):
            story.append(Paragraph(inline(payload), st[kind]))
        else:
            story.append(Paragraph(inline(payload), st["body"]))

    doc = SimpleDocTemplate(
        str(TARGET), pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.8 * inch, bottomMargin=0.85 * inch,
        title="Freight Rate Prediction", author="",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(f"wrote {TARGET}  ({TARGET.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
