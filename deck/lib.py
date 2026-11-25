"""Small helper library for building the deck with python-pptx.

Design system: 16:9, white background, Times New Roman everywhere, black text
and a ladder of greys.  No colour accents, no decorative bars.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FONT = "Times New Roman"
W, H = 13.333, 7.5
MX = 0.75                      # left/right margin
CW = W - 2 * MX                # content width

INK, G1, G2, G3, G4, G5, G6, G7 = (
    "111111", "2b2b2b", "4d4d4d", "6e6e6e", "8f8f8f", "bdbdbd", "e3e3e3", "f3f3f3")

EQ_DIR = Path("deck/_eq")
EQ_DIR.mkdir(parents=True, exist_ok=True)


def n_images() -> int:
    """Number of images actually analysed (from results/summary.json)."""
    import json
    f = Path("results/summary.json")
    if f.exists():
        return int(json.loads(f.read_text())["n_images"])
    return 20


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h.upper())


# ------------------------------------------------------------------ deck
def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    # theme fonts -> Times New Roman, so anything not set explicitly matches
    theme_part = None
    for rel in prs.slide_masters[0].part.rels.values():
        if rel.reltype.endswith("/theme"):
            theme_part = rel.target_part
    if theme_part is not None:
        xml = theme_part.blob.decode("utf8")
        xml = re.sub(r'(<a:latin typeface=")[^"]*(")', r"\1" + FONT + r"\2", xml)
        theme_part._blob = xml.encode("utf8")
    return prs


def blank(prs) -> "Slide":
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb("FFFFFF")
    return s


# ------------------------------------------------------------------ text
_BOLD = re.compile(r"(\*\*.+?\*\*)")


def _runs(p, text: str, size, color, bold, italic, markup=True):
    for part in (_BOLD.split(text) if markup else [text]):
        if not part:
            continue
        b = markup and part.startswith("**") and part.endswith("**")
        r = p.add_run()
        r.text = part[2:-2] if b else part
        f = r.font
        f.name, f.size, f.bold, f.italic = FONT, Pt(size), bool(bold or b), italic
        f.color.rgb = rgb(color)
        rPr = r._r.get_or_add_rPr()
        for tag in ("a:ea", "a:cs"):
            el = etree.SubElement(rPr, qn(tag))
            el.set("typeface", FONT)


def _bullet(p, char="•", indent=0.22, level_in=0.0):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent + level_in))))
    pPr.set("indent", str(-int(Inches(indent))))
    for tag in ("a:buNone", "a:buChar", "a:buFont"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    bf = etree.SubElement(pPr, qn("a:buFont"))
    bf.set("typeface", FONT)
    bc = etree.SubElement(pPr, qn("a:buChar"))
    bc.set("char", char)


def tb(slide, x, y, w, h, paras, size=16, color=INK, bold=False, italic=False, align="l",
       anchor="t", line=1.08, after=6, name=None):
    """Add a text box.  `paras` is a str or a list of str / dict(text, size, bold, color,
    italic, bullet, after, align).  **bold** inline markup is supported."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        d = {"text": item} if isinstance(item, str) else dict(item)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[d.get("align", align)]
        p.line_spacing = d.get("line", line)
        p.space_after = Pt(d.get("after", after))
        p.space_before = Pt(0)
        _runs(p, d["text"], d.get("size", size), d.get("color", color), d.get("bold", bold),
              d.get("italic", italic), markup=not d.get("raw", False))
        if d.get("bullet"):
            _bullet(p, level_in=0.25 * d.get("level", 0))
    return box


def bullets(slide, x, y, w, h, items, size=16, color=INK, after=8, line=1.08):
    return tb(slide, x, y, w, h, [{"text": t, "bullet": True} if isinstance(t, str) else {**t, "bullet": True}
                                  for t in items], size=size, color=color, after=after, line=line)


# ---------------------------------------------------------------- shapes
def card(slide, x, y, w, h, fill=G7, line=None, radius=0.04):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(0.75)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def box_text(slide, x, y, w, h, text, size=15, fill=G7, line=None, color=INK, bold=False, align="c",
             pad=0.12, radius=0.06):
    shp = card(slide, x, y, w, h, fill=fill, line=line, radius=radius)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(pad)
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    paras = text if isinstance(text, list) else [text]
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[align]
        p.space_after = Pt(2)
        _runs(p, t, size, color, bold, False)
    return shp


def arrow(slide, x1, y1, x2, y2, color=G3, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def hline(slide, x, y, w, color=G6, width=0.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + w), Inches(y))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    return c


# --------------------------------------------------------------- frames
_state = {"n": 0}


def header(slide, kicker: str, title: str, size=30):
    tb(slide, MX, 0.48, CW, 0.3, kicker.upper(), size=12, color=G3, bold=False, after=0, name="kicker")
    tb(slide, MX, 0.82, CW, 1.0, title, size=size, bold=True, color=INK, line=1.0, after=0, name="title")


def footer(slide, n: int, label="Adversarial attacks on explainable AI  |  image classification"):
    tb(slide, MX, 7.02, 8, 0.25, label, size=10, color=G4, after=0, name="footer")
    tb(slide, W - MX - 1.0, 7.02, 1.0, 0.25, str(n), size=10, color=G4, align="r", after=0, name="page")


def notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- images
def pic(slide, path, x, y, w=None, h=None, align="c"):
    """Insert an image fitted inside the (x,y,w,h) box, preserving aspect ratio."""
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    if w and h:
        if w / h > ar:
            nw, nh = h * ar, h
        else:
            nw, nh = w, w / ar
        ox = x + (w - nw) / 2 if align == "c" else x
        oy = y + (h - nh) / 2
    elif w:
        nw, nh, ox, oy = w, w / ar, x, y
    else:
        nw, nh, ox, oy = h * ar, h, x, y
    return slide.shapes.add_picture(str(path), Inches(ox), Inches(oy), Inches(nw), Inches(nh))


def eq(slide, latex: str, x, y, w=None, size=26, align="l", max_w=None):
    """Render LaTeX (matplotlib mathtext, STIX = Times look) and place it.
    Returns (width, height) in inches."""
    key = hashlib.md5(f"{latex}|{size}".encode()).hexdigest()[:12]
    path = EQ_DIR / f"{key}.png"
    if not path.exists():
        plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, latex, fontsize=size, color="#" + INK)
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    with Image.open(path) as im:
        iw, ih = im.size
    ew, eh = iw / 300, ih / 300
    lim = max_w or w
    if lim and ew > lim:
        eh, ew = eh * lim / ew, lim
    px = x
    if w and align == "c":
        px = x + (w - ew) / 2
    elif w and align == "r":
        px = x + w - ew
    slide.shapes.add_picture(str(path), Inches(px), Inches(y), Inches(ew), Inches(eh))
    return ew, eh


# ----------------------------------------------------------------- table
def _cell_border(cell, sides=("b",), color=G5, width=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    for s in ("l", "r", "t", "b"):
        tag = qn(f"a:ln{s.upper()}")
        for el in tcPr.findall(tag):
            tcPr.remove(el)
    for s in ("l", "r", "t", "b"):
        ln = etree.SubElement(tcPr, qn(f"a:ln{s.upper()}"))
        if s in sides:
            ln.set("w", str(int(Pt(width))))
            sf = etree.SubElement(ln, qn("a:solidFill"))
            c = etree.SubElement(sf, qn("a:srgbClr"))
            c.set("val", color)
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
    # a:ln* must precede fill elements in tcPr
    fills = [e for e in tcPr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]
    for e in fills:
        tcPr.remove(e)
        tcPr.append(e)


def table(slide, x, y, w, rows, col_w=None, size=13, row_h=0.42, head_size=None, align=None,
          bold_first_col=False, hdr_fill=G7, head_h=None):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("bandRow", "firstRow", "bandCol", "firstCol", "lastRow", "lastCol"):
        tblPr.set(a, "0")
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    col_w = col_w or [1 / nc] * nc
    tot = sum(col_w)
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(w * cw / tot)
    for r in range(nr):
        tbl.rows[r].height = Inches(head_h if (r == 0 and head_h) else row_h)
        for c in range(nc):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            a = (align[c] if align else "l")
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[a]
            hdr = r == 0
            _runs(p, str(rows[r][c]), (head_size or size) if hdr else size, INK,
                  hdr or (bold_first_col and c == 0), False)
            if hdr:
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(hdr_fill)
            else:
                cell.fill.background()
            _cell_border(cell, sides=("b",), color=G5 if hdr else G6, width=1.0 if hdr else 0.5)
    return gf
