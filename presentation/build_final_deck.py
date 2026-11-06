"""
build_final_deck.py -- build the final presentation from the midsem deck.

It opens presentation/midsem_tulu_hope_speech.pptx (never changed), updates the slides
whose content changed, adds new slides in the same design (M4, combinations, test set,
analysis, takeaways, limitations), puts them in order and saves
presentation/final_tulu_hope_speech.pptx.

Every number on the slides comes from results/ (phase4_summary.md, phase5_summary.md,
combine_dev.csv, test_scores.csv, test_parts.csv).

Run from the main project folder (tulu-hope-speech):
    python presentation/build_final_deck.py

Requires: python-pptx  (pip install python-pptx)
"""

import copy

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

SRC = "presentation/midsem_tulu_hope_speech.pptx"
OUT = "presentation/final_tulu_hope_speech.pptx"

# The midsem deck's colours
DARK = "0E4D5A"    # dark teal (title slide, table header)
TEAL = "1F7A8C"
AMBER = "F2A93B"
RED = "D1495B"
INK = "1F2933"     # main text
MUTED = "5F6B76"   # secondary text
CARD = "E6F2F4"    # light teal card
CARD_AMBER = "FDF0DC"
PALE = "C9E3E8"
GRID = "DDE3E6"
WHITE = "FFFFFF"


# ----------------------------------------------------------------- small helpers

def rgb(hex_color):
    return RGBColor.from_string(hex_color)


def run(text, size, bold=False, color=INK, font="Arial"):
    """One piece of text with one style. Used to build paragraphs."""
    return {"text": text, "size": size, "bold": bold, "color": color, "font": font}


def add_text(slide, x, y, w, h, paragraphs, anchor="t", align="l", after=0):
    """A text box like the midsem deck's: no inner margins, grows to fit its text.
    paragraphs: a list; each paragraph is a list of run(...) pieces."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[anchor]
    for i, pieces in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if after and i < len(paragraphs) - 1:
            p.space_after = Pt(after)
        for piece in pieces:
            r = p.add_run()
            r.text = piece["text"]
            r.font.size = Pt(piece["size"])
            r.font.bold = piece["bold"]
            r.font.italic = False
            r.font.color.rgb = rgb(piece["color"])
            r.font.name = piece["font"]
    return box


def add_box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    """A filled card (rounded rectangle) or circle, without shadow."""
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def badge(slide, x, y, d, label, fill, text_color, size):
    """A numbered circle."""
    add_box(slide, x, y, d, d, fill, shape=MSO_SHAPE.OVAL)
    add_text(slide, x, y, d, d, [[run(label, size, True, text_color)]], anchor="m", align="c")


def heading_and_body(slide, x, y, w, h, heading, body, hsize=15, bsize=12):
    """Bold heading with a muted line (or lines) under it, vertically centred."""
    paras = [[run(heading, hsize, True, INK)]]
    for line in ([body] if isinstance(body, str) else body):
        paras.append([run(line, bsize, False, MUTED)])
    return add_text(slide, x, y, w, h, paras, anchor="m", after=3)


def header(slide, kicker, title):
    add_text(slide, 0.6, 0.32, 12.1, 0.35, [[run(kicker, 12, True, TEAL)]])
    add_text(slide, 0.6, 0.62, 12.1, 0.8, [[run(title, 30, True, INK, "Georgia")]])


def caption(slide, x, y, w, text, size=10):
    add_text(slide, x, y, w, 0.3, [[run(text, size, False, MUTED)]])


def arrow(slide, x1, y1, x2, y2, color=TEAL):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                      Inches(x2), Inches(y2))
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(2)
    ln = line.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return line


def style_chart(chart, colors, vmin, vmax, legend=True, label_format="0.000",
                cat_size=12, label_size=11, value_axis=True):
    """Make a chart look like the midsem charts: no title, value labels, light grid."""
    chart.has_title = False
    chart.font.name = "Arial"
    chart.font.size = Pt(11)
    chart.font.color.rgb = rgb(INK)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
    plot = chart.plots[0]
    plot.gap_width = 70
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = label_format
    labels.number_format_is_linked = False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(label_size)
    labels.font.color.rgb = rgb(INK)
    for series, color in zip(plot.series, colors):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = rgb(color)
        series.format.line.fill.background()
    cat = chart.category_axis
    cat.tick_labels.font.size = Pt(cat_size)
    cat.tick_labels.font.color.rgb = rgb(INK)
    cat.format.line.color.rgb = rgb(GRID)
    cat.has_major_gridlines = False
    val = chart.value_axis
    val.minimum_scale = vmin
    val.maximum_scale = vmax
    val.has_major_gridlines = True
    val.major_gridlines.format.line.color.rgb = rgb(GRID)
    val.major_gridlines.format.line.width = Pt(0.75)
    val.tick_labels.number_format = "0.0" if label_format == "0.000" else "0"
    val.tick_labels.number_format_is_linked = False
    val.tick_labels.font.size = Pt(10)
    val.tick_labels.font.color.rgb = rgb(MUTED)
    val.format.line.fill.background()
    val.visible = value_axis
    return plot


def shape_named(slide, name):
    for s in slide.shapes:
        if s.name == name:
            return s
    raise KeyError(name)


def set_runs(slide, name, texts):
    """Replace the text of a shape's runs, keeping their formatting.
    texts: one string per run, in reading order."""
    runs = [r for p in shape_named(slide, name).text_frame.paragraphs for r in p.runs]
    assert len(runs) == len(texts), (name, [r.text for r in runs])
    for r, t in zip(runs, texts):
        r.text = t


def set_fill(slide, name, color):
    s = shape_named(slide, name)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(color)
    s.line.fill.background()


def set_run_color(slide, name, color):
    for p in shape_named(slide, name).text_frame.paragraphs:
        for r in p.runs:
            r.font.color.rgb = rgb(color)


def remove(shape):
    shape._element.getparent().remove(shape._element)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def results_table(slide, x, y, col_widths, row_h, rows, highlight=(), size=12, left_cols=2):
    """A table like the midsem results table: dark header, alternating light rows.
    rows[0] is the header. highlight: set of (row, col) cells to fill amber."""
    shape = slide.shapes.add_table(len(rows), len(col_widths), Inches(x), Inches(y),
                                   Inches(sum(col_widths)), Inches(row_h * len(rows)))
    table = shape.table
    for j, w in enumerate(col_widths):
        table.columns[j].width = Inches(w)
    for i, values in enumerate(rows):
        table.rows[i].height = Inches(row_h)
        for j, value in enumerate(values):
            cell = table.cell(i, j)
            if i == 0:
                fill, color, bold = DARK, WHITE, True
            elif (i, j) in highlight:
                fill, color, bold = AMBER, INK, True
            else:
                fill, color, bold = (CARD if i % 2 == 1 else WHITE), INK, False
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j < left_cols else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = value
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = False
            r.font.color.rgb = rgb(color)
            r.font.name = "Arial"
    return shape


# ----------------------------------------------------------------- updated slides

def update_title(s):
    set_runs(s, "TextBox 1", ["IT702 DEEP LEARNING  ·  FINAL EVALUATION"])
    set_runs(s, "TextBox 3", ["Final report: baselines, character models, MuRIL, "
                              "a pair head, ensembles and the test set"])
    set_runs(s, "TextBox 4", ["NITK Surathkal  ·  November 2026"])
    notes(s, "Introduce the project: we classify YouTube-style comments written in "
             "code-mixed Tulu. Each comment gets two labels at the same time: its tone "
             "(3 classes) and its type of hope (5 classes). The example on the right is a "
             "real training comment. This is the final presentation: all models are "
             "trained and the test set has been scored once. Add your team names here.")


def update_overview(s):
    set_runs(s, "TextBox 2", ["Final status: every planned experiment is done"])
    set_runs(s, "TextBox 6", ["Data repair and cleaning",
                              "test labels fixed; 148 noisy training rows removed"])
    set_runs(s, "TextBox 9", ["M1 · TF-IDF baseline", "dev: tone 0.579, hope type 0.400"])
    set_runs(s, "TextBox 18", ["M4 · one head over the 14 tone–type pairs",
                               "10 models; over-predicted tiny pairs fixed with new weights"])
    set_runs(s, "TextBox 21", ["Combinations, then the test set (once)",
                               "final systems fixed on dev; 25 prediction files scored"])
    set_runs(s, "TextBox 24", ["Why test scores are lower",
                               "one part of the test file has unreliable labels"])
    # Steps 5-7 are done now: same filled circles as steps 1-4.
    for oval, label, item in (("Oval 16", "TextBox 17", "TextBox 18"),
                              ("Oval 19", "TextBox 20", "TextBox 21"),
                              ("Oval 22", "TextBox 23", "TextBox 24")):
        set_fill(s, oval, DARK)
        set_run_color(s, label, WHITE)
        first = shape_named(s, item).text_frame.paragraphs[0].runs[0]
        first.font.color.rgb = rgb(INK)
    set_runs(s, "TextBox 26", ["55"])
    set_runs(s, "TextBox 27", ["neural models trained",
                               "3 models × 3 tasks × 5 seeds, plus 2 M4 runs × 5 seeds"])
    set_runs(s, "TextBox 29", ["0.437"])
    set_runs(s, "TextBox 30", ["best dev hope-type F1", "M4b pair head, vs 0.400 for M1"])
    set_runs(s, "TextBox 32", ["0.364"])
    set_runs(s, "TextBox 33", ["test Joint Score",
                               "main system: M3 multi-task, 5 seeds averaged"])
    notes(s, "All seven parts of the plan are done. After the midsem we built M4, a single "
             "head over the 14 tone and type pairs, combined models on the dev set, fixed "
             "the final systems, and scored the test set once. The test score is lower than "
             "dev, and we found why: one part of the test file has labels that look "
             "unreliable.")


def update_results_table(s):
    old = [sh for sh in s.shapes if sh.has_table][0]
    remove(old)
    rows = [
        ["Model", "Training", "Tone F1", "Type F1", "Joint"],
        ["M1 · TF-IDF + LR", "separate", "0.579", "0.400", "0.490"],
        ["M2 · CNN", "multi-task", "0.547 ± 0.010", "0.411 ± 0.009", "0.479"],
        ["M2 · CNN", "single", "0.558 ± 0.013", "0.409 ± 0.012", "(0.484)"],
        ["M2 · BiLSTM", "multi-task", "0.513 ± 0.011", "0.382 ± 0.012", "0.448"],
        ["M2 · BiLSTM", "single", "0.523 ± 0.023", "0.375 ± 0.012", "(0.449)"],
        ["M3 · MuRIL", "multi-task", "0.540 ± 0.015", "0.426 ± 0.013", "0.483"],
        ["M3 · MuRIL", "single", "0.551 ± 0.015", "0.434 ± 0.015", "(0.493)"],
        ["M4 · MuRIL", "pair head", "0.521 ± 0.012", "0.418 ± 0.007", "0.469"],
        ["M4b · MuRIL", "pair head v2", "0.523 ± 0.011", "0.437 ± 0.009", "0.480"],
    ]
    results_table(s, 0.6, 1.65, [2.1, 1.45, 1.6, 1.6, 1.0], 0.44, rows,
                  highlight={(1, 2), (9, 3)})
    set_runs(s, "TextBox 2", ["All models side by side (dev)"])
    set_runs(s, "TextBox 5", ["Dev macro F1, mean ± std over 5 seeds. Joint in brackets = the "
                              "tone-only and type-only models used together. Highlighted = "
                              "best tone and best hope type."])
    set_runs(s, "TextBox 13", ["Hope type", "M4b: 0.437 vs 0.400 for M1"])
    set_runs(s, "TextBox 17", ["Joint", "the top models are within 0.015 of each other: a tie"])
    notes(s, "All models on the dev set. M1 is still best on tone. On hope type, the second "
             "M4 run (M4b) is now best, at 0.437, level with single-task MuRIL. On the joint "
             "score the top models are within about one and a half points of each other, "
             "which is within the seed-to-seed spread.")


def update_per_class(s):
    chart = [sh for sh in s.shapes if sh.has_chart][0].chart
    data = CategoryChartData()
    data.categories = ["M1", "CNN\nmulti", "BiLSTM\nmulti", "MuRIL\nmulti", "MuRIL\nsingle",
                       "M4b\npairs"]
    data.add_series("Fading Hope F1", (0.206, 0.214, 0.175, 0.278, 0.302, 0.302))
    data.add_series("Realistic Hope F1", (0.235, 0.252, 0.178, 0.238, 0.233, 0.219))
    chart.replace_data(data)
    set_runs(s, "TextBox 7", ["Fading Hope: single-task MuRIL and M4b both reach 0.302, vs "
                              "0.206 for M1 (57 dev comments)"])
    set_runs(s, "TextBox 12", ["0 of 55"])
    notes(s, "Macro F1 rewards getting rare classes right. MuRIL's gain over M1 comes mostly "
             "from Fading Hope, the rarest class, where it improves F1 by almost half. M4b "
             "matches the best Fading Hope score. Realistic Hope remains hard for every "
             "model, which points to overlapping labels rather than a weak model.")


def update_protocol(s):
    """The midsem 'plan' slide becomes the final evaluation protocol (same timeline)."""
    set_runs(s, "TextBox 1", ["FINAL EVALUATION"])
    set_runs(s, "TextBox 2", ["How the final systems were chosen and scored"])
    steps = [
        ("Probabilities", "every model saves class probabilities for dev and the unlabelled "
                          "test file"),
        ("Combine on dev", "seed averages and 11 equal-weight combinations, scored on dev "
                           "only"),
        ("Declare", "main: M3 multi-task; best overall: all four combined"),
        ("Commit", "test predictions saved in git before the test labels were used"),
        ("Score once", "score_test.py on all 25 prediction files"),
    ]
    for (title, desc), (t, d) in zip(steps, [("TextBox 7", "TextBox 8"),
                                             ("TextBox 11", "TextBox 12"),
                                             ("TextBox 15", "TextBox 16"),
                                             ("TextBox 19", "TextBox 20"),
                                             ("TextBox 23", "TextBox 24")]):
        set_runs(s, t, [title])
        set_runs(s, d, [desc])
    # Highlight the last step (the one test score) instead of the first.
    set_fill(s, "Oval 5", DARK)
    set_run_color(s, "TextBox 6", WHITE)
    set_fill(s, "Oval 21", AMBER)
    set_run_color(s, "TextBox 22", INK)
    set_runs(s, "TextBox 26", ["Why equal weights only:  ",
                               "tuning one weight per model on 753 dev comments would fit the "
                               "dev set itself."])
    notes(s, "This is how we kept the final test score honest. Every model saved its "
             "probabilities. We tried seed averaging and combinations on the dev set only, "
             "then declared the final systems: M3 multi-task as the main system and all four "
             "models combined as the best overall. The test predictions were committed to git "
             "before we used the test labels, and then all of them were scored once.")


def update_appendix(s):
    box = shape_named(s, "TextBox 5")
    tf = box.text_frame
    template = copy.deepcopy(tf.paragraphs[0]._p)       # a comment line, for its style
    lines = [
        ("# all saved results in one table (instant)", True),
        ("python scripts/show_results.py", False),
        ("", False),
        ("# M1 baseline from scratch (about 5 s)", True),
        ("python scripts/explore_multitask.py \\", False),
        ("    --train data/train_clean.csv \\", False),
        ("    --dev data/dev_multitask.csv", False),
        ("", False),
        ("# combinations on dev (instant)", True),
        ("python scripts/combine_dev.py", False),
        ("", False),
        ("# final test scores (scored once; only read here)", True),
        ("cat results/test_scores.csv", False),
        ("", False),
        ("# why test is lower than dev (instant)", True),
        ("python scripts/analyse_test.py", False),
    ]
    for p in list(tf.paragraphs):
        p._p.getparent().remove(p._p)
    for text, is_comment in lines:
        p = copy.deepcopy(template)
        tf._txBody.append(p)
        r = p.findall(qn("a:r"))[0]
        r.find(qn("a:t")).text = text
        color = r.find(qn("a:rPr")).find(qn("a:solidFill")).find(qn("a:srgbClr"))
        color.set("val", MUTED if is_comment else WHITE)
    set_runs(s, "TextBox 9", ["GPU runs", "BiLSTM, MuRIL and M4 ran on Kaggle "
                              "(notebooks/*_kaggle.ipynb); final reruns: "
                              "final_probs_kaggle.ipynb"])
    set_runs(s, "TextBox 11", ["Saved", "scores and settings in results/*.json; summaries in "
                               "results/phase*_summary.md; history in git"])
    notes(s, "If asked to show the code: run these from the tulu-hope-speech folder after "
             "activating the environment. None of them trains anything or re-scores the test "
             "set: they read saved results.")


# ----------------------------------------------------------------- new slides

def slide_m4_idea(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "M4 · ONE HEAD OVER PAIRS", "M4: predict the (tone, type) pair directly")
    boxes = [
        ("Comment", "“baari porlu movie”", CARD),
        ("MuRIL encoder", "same as M3: mean pooling → BatchNorm → dropout", CARD),
        ("Pair head", "one linear layer: 768 numbers → 14 pairs", AMBER),
        ("Split the pair", "gives one tone and one type, scored like M1–M3", CARD),
    ]
    x, w, gap = 0.6, 2.7, 0.433
    for i, (title, desc, fill) in enumerate(boxes):
        bx = x + i * (w + gap)
        add_box(s, bx, 1.7, w, 1.35, fill)
        heading_and_body(s, bx + 0.2, 1.8, w - 0.4, 1.15, title, desc, hsize=15, bsize=12)
        if i < 3:
            arrow(s, bx + w + 0.06, 2.375, bx + w + gap - 0.06, 2.375)

    # Left card: why
    add_box(s, 0.6, 3.4, 5.95, 3.4, CARD)
    add_text(s, 0.85, 3.55, 5.5, 3.1, [
        [run("Why one head instead of two?", 15, True, INK)],
        [run("M3's two heads choose tone and type separately, so they can output a pair "
             "that never occurs, such as Discouraging + Optimistic.", 13, False, MUTED)],
        [run("One head over the 14 pairs seen in training can only output real pairs, and "
             "it learns tone and type as one decision.", 13, False, MUTED)],
        [run("Training as in M3: same MuRIL settings, class-weighted loss over the pairs, "
             "best epoch by dev Joint Score, 5 seeds.", 13, False, MUTED)],
        [run("The risk: with one output per pair, a rare pair has almost nothing to learn "
             "from (right).", 13, True, INK)],
    ], anchor="t", after=8)

    # Right card: the tiny pairs
    add_box(s, 6.75, 3.4, 5.95, 3.4, CARD_AMBER)
    add_text(s, 7.0, 3.55, 5.5, 0.4, [[run("The catch: four pairs are tiny", 15, True, INK)]])
    rows = [["Pair", "Train", "Dev"],
            ["Encouraging + Fading Hope", "1", "2"],
            ["Encouraging + Hopelessness", "1", "0"],
            ["Discouraging + Inspiring", "3", "2"],
            ["Discouraging + Realistic", "3", "0"]]
    results_table(s, 7.0, 4.05, [3.5, 1.0, 1.0], 0.42, rows, size=12, left_cols=1)
    caption(s, 7.0, 6.25, 5.5, "Out of 3,365 training comments. The largest pair "
                               "(Encouraging + Inspiring) has 1,105.", size=11)
    notes(s, "M4 changes only the output of M3. Instead of two heads, one for tone and one "
             "for hope type, it has a single head with one output for each of the 14 pairs "
             "that occur in training. The predicted pair is then split into a tone and a "
             "type, so it is scored exactly like the other models. The catch: four of the "
             "pairs have only one to three training comments.")
    return s


def slide_m4_results(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "M4 · RESULTS", "The first M4 run failed on tiny pairs; new weights fixed it")
    cards = [
        ("1", RED, WHITE, CARD, "First run: tiny pairs over-predicted",
         "25–40 dev comments per seed went to the four tiny pairs, and none were right. "
         "A 1-comment pair had about 33× the loss weight of the largest pair."),
        ("2", TEAL, WHITE, CARD, "Fix: weight = tone weight × type weight",
         "Tiny pairs drop to about 1–2×; the Fading Hope pairs get the most (about 3.5×). "
         "Also 8 epochs instead of 5. Chosen on dev."),
        ("3", AMBER, INK, CARD_AMBER, "Result: M4b",
         "No tiny-pair predictions. Best dev hope type of any model (0.437), but tone "
         "(0.523) stays below M3 (0.540)."),
    ]
    for i, (num, fill, tcol, card, title, body) in enumerate(cards):
        y = 1.65 + i * 1.75
        add_box(s, 0.6, y, 6.3, 1.55, card)
        badge(s, 0.8, y + 0.45, 0.65, num, fill, tcol, 20)
        heading_and_body(s, 1.65, y + 0.12, 5.1, 1.35, title, body)

    data = CategoryChartData()
    data.categories = ["M1", "M3\ntwo heads", "M4\nfirst run", "M4b\nnew weights"]
    data.add_series("Tone macro F1", (0.579, 0.540, 0.521, 0.523))
    data.add_series("Hope-type macro F1", (0.400, 0.426, 0.418, 0.437))
    frame = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.2), Inches(1.55),
                               Inches(5.6), Inches(4.9), data)
    plot = style_chart(frame.chart, [DARK, AMBER], 0.0, 0.7)
    plot.overlap = -10
    caption(s, 7.3, 6.5, 5.4, "Dev macro F1, mean of 5 seeds (M1: single run). "
                              "M3 = multi-task, two heads.")
    notes(s, "Our first M4 run was worse than M3. It kept predicting the four tiny pairs, "
             "and those predictions were always wrong. The cause was the loss weights: a pair "
             "with one training comment got about 33 times the weight of the biggest pair. "
             "We changed the weights to tone weight times type weight, because the metric is "
             "computed per tone and per type, not per pair, and trained for 8 epochs. The "
             "second run, M4b, never predicts the tiny pairs and has the best hope-type score "
             "of any model, but its tone score is lower than M3's.")
    return s


def slide_combinations(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "COMBINING MODELS", "Averaging models helps on dev, by about 0.01")
    # (label, dev Joint, colour). Source: results/combine_dev.csv ("all seeds" rows).
    rows = [
        ("M4b", 0.479, PALE), ("M1", 0.489, PALE), ("CNN", 0.492, PALE), ("M3", 0.493, PALE),
        ("M3 + M4b", 0.492, TEAL), ("M1 + M3", 0.495, TEAL), ("M1 + CNN + M3", 0.499, TEAL),
        ("CNN + M3 + M4b", 0.499, TEAL), ("M1 + CNN", 0.500, TEAL), ("CNN + M3", 0.500, TEAL),
        ("CNN + M4b", 0.503, TEAL), ("All four", 0.504, AMBER), ("M1 + CNN + M4b", 0.506, TEAL),
        ("M1 + M3 + M4b", 0.507, TEAL), ("M1 + M4b", 0.512, TEAL),
    ]
    # A horizontal bar chart draws the first category at the bottom, so reverse.
    rows = rows[::-1]
    data = CategoryChartData()
    data.categories = [r[0] for r in rows]
    data.add_series("Dev Joint Score", [r[1] for r in rows])
    frame = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.5), Inches(1.5),
                               Inches(7.9), Inches(5.35), data)
    plot = style_chart(frame.chart, [TEAL], 0.40, 0.53, legend=False, cat_size=11,
                       label_size=10)
    plot.gap_width = 40
    frame.chart.value_axis.major_unit = 0.05
    frame.chart.value_axis.tick_labels.number_format = "0.00"
    series = plot.series[0]
    for i, (_, _, color) in enumerate(rows):
        point = series.points[i]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = rgb(color)
    caption(s, 0.6, 6.95, 7.8, "Dev Joint Score. Light = one model (5 seeds averaged); "
                               "dark = combination, equal weights. Axis starts at 0.40.")

    add_box(s, 8.8, 1.6, 3.9, 1.6, CARD_AMBER)
    add_text(s, 9.0, 1.72, 3.6, 0.6, [[run("0.504", 26, True, INK, "Georgia")]])
    add_text(s, 9.0, 2.32, 3.6, 0.85, [[run("All four combined, vs 0.493 for the best single "
                                            "model (M3, 5 seeds averaged)", 12, False, MUTED)]])
    add_box(s, 8.8, 3.4, 3.9, 1.6, CARD)
    add_text(s, 9.0, 3.52, 3.6, 0.6, [[run("11 tried", 26, True, DARK, "Georgia")]])
    add_text(s, 9.0, 4.12, 3.6, 0.85, [[run("The best, M1 + M4b (0.512), wins partly by luck: "
                                            "bootstrap 95% ranges of the top ones overlap",
                                            12, False, MUTED)]])
    add_box(s, 8.8, 5.2, 3.9, 1.6, CARD)
    add_text(s, 9.0, 5.32, 3.6, 0.6, [[run("Declared", 26, True, DARK, "Georgia")]])
    add_text(s, 9.0, 5.92, 3.6, 0.85, [[run("Main: M3 multi-task (one forward pass). Best "
                                            "overall: all four, no picking.", 12, False, MUTED)]])
    notes(s, "Each bar is a dev Joint Score. The light bars are single models with their five "
             "seeds averaged; the dark bars are combinations, where we average the "
             "probabilities of the models with equal weights. Every combination with M1 beats "
             "every single model, but only by about one point. M1 plus M4b looks best, but it "
             "was the best of 11 tries on the same 753 comments, and a bootstrap shows the top "
             "combinations overlap. So we declared M3 as the main system, because it is one "
             "model with one forward pass, and all four combined as the best overall, because "
             "it needs no picking.")
    return s


def slide_test(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "TEST SET", "On test, M3 multi-task is the best system")
    data = CategoryChartData()
    data.categories = ["M3\nmain", "All four\ncombined", "CNN\nmulti", "M4b\npairs", "M1\nTF-IDF"]
    data.add_series("Dev Joint", (0.493, 0.504, 0.492, 0.479, 0.489))
    data.add_series("Test Joint", (0.364, 0.341, 0.352, 0.343, 0.326))
    frame = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.55),
                               Inches(7.9), Inches(5.3), data)
    plot = style_chart(frame.chart, [PALE, DARK], 0.0, 0.6)
    plot.overlap = -10
    caption(s, 0.6, 6.95, 7.8, "Joint Score, 5 seeds averaged (M1: single run). Test scored "
                               "once, after the systems were fixed on dev.")

    add_box(s, 8.8, 1.6, 3.9, 1.6, CARD_AMBER)
    add_text(s, 9.0, 1.72, 3.6, 0.6, [[run("0.364", 26, True, INK, "Georgia")]])
    add_text(s, 9.0, 2.32, 3.6, 0.85, [[run("M3 test Joint (main system). Per seed: "
                                            "0.350 ± 0.003, the smallest spread", 12, False,
                                            MUTED)]])
    add_box(s, 8.8, 3.4, 3.9, 1.6, CARD)
    add_text(s, 9.0, 3.52, 3.6, 0.6, [[run("–0.023", 26, True, DARK, "Georgia")]])
    add_text(s, 9.0, 4.12, 3.6, 0.85, [[run("All four combined vs M3 on test; it was +0.011 "
                                            "on dev. Combining did not carry over", 12, False,
                                            MUTED)]])
    add_box(s, 8.8, 5.2, 3.9, 1.6, CARD)
    add_text(s, 9.0, 5.32, 3.6, 0.6, [[run("0.489 → 0.326", 26, True, DARK, "Georgia")]])
    add_text(s, 9.0, 5.92, 3.6, 0.85, [[run("M1 falls the most: from the best single model "
                                            "on dev (per seed) to last on test", 12, False,
                                            MUTED)]])
    notes(s, "These are the test scores, computed once. Every system drops by 0.13 to 0.16 "
             "from dev to test. M3 multi-task, our declared main system, is the best on test, "
             "with 0.364 when its seeds are averaged and the smallest spread across seeds. "
             "Combining models did not carry over to test: all four combined is now below M3. "
             "M1 drops the most, which pulls the combinations down.")
    return s


def slide_test_parts(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "TEST SET · ANALYSIS", "Most of the drop comes from one part of the test file")
    data = CategoryChartData()
    data.categories = ["Test rows 429–744", "Test rows 1–428", "Dev", "Train"]
    data.add_series("Encouraging", (42.4, 56.8, 56.6, 56.6))
    data.add_series("Discouraging", (55.7, 22.0, 21.8, 21.2))
    data.add_series("Blended", (1.9, 21.3, 21.6, 22.2))
    frame = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(0.5), Inches(1.5),
                               Inches(6.6), Inches(3.1), data)
    chart = frame.chart
    plot = style_chart(chart, [DARK, RED, AMBER], 0, 100, legend=True, label_format='0"%"',
                       cat_size=11, value_axis=False)
    plot.gap_width = 45
    plot.overlap = 100
    plot.data_labels.position = XL_LABEL_POSITION.CENTER
    for series, color in zip(plot.series, (WHITE, WHITE, INK)):
        series.data_labels.show_value = True
        series.data_labels.font.color.rgb = rgb(color)
        series.data_labels.font.size = Pt(11)
        series.data_labels.number_format = '0"%"'
        series.data_labels.number_format_is_linked = False
        series.data_labels.position = XL_LABEL_POSITION.CENTER
    # The 2% Blended label does not fit inside its bar: delete that one label
    # (a <c:dLbl> with <c:delete val="1"/> for point 0, placed first in <c:dLbls>).
    labels_xml = plot.series[2].data_labels._element
    hidden = etree.Element(qn("c:dLbl"))
    etree.SubElement(hidden, qn("c:idx")).set("val", "0")
    etree.SubElement(hidden, qn("c:delete")).set("val", "1")
    labels_xml.insert(0, hidden)
    chart.value_axis.has_major_gridlines = False
    caption(s, 0.6, 4.6, 6.4, "Tone mix (% of comments). Rows 429–744 are the block whose "
                              "label columns were swapped in the released file.")

    for i, (title, joint, acc, fill, note) in enumerate([
            ("Rows 1–428 (428 comments)", "0.410", "0.55", CARD,
             "label mix like train and dev"),
            ("Rows 429–744 (316 comments)", "0.298", "0.33", CARD_AMBER,
             "tone accuracy at chance (3 tones)")]):
        x = 7.4 + i * 2.7
        add_box(s, x, 1.55, 2.55, 3.25, fill)
        add_text(s, x + 0.18, 1.7, 2.2, 0.6, [[run(title, 13, True, INK)]])
        add_text(s, x + 0.18, 2.3, 2.2, 0.6, [[run(joint, 26, True, DARK, "Georgia")]])
        add_text(s, x + 0.18, 2.9, 2.2, 0.3, [[run("M3 Joint Score", 12, False, MUTED)]])
        add_text(s, x + 0.18, 3.3, 2.2, 0.6, [[run(acc, 26, True, DARK, "Georgia")]])
        add_text(s, x + 0.18, 3.9, 2.2, 0.8, [[run("M3 tone accuracy; " + note, 12, False,
                                                   MUTED)]])

    add_box(s, 0.6, 5.05, 12.1, 1.75, CARD)
    add_text(s, 0.85, 5.17, 11.6, 1.55, [
        [run("Labels in rows 429–744 look unreliable, not shifted", 15, True, INK)],
        [run("“edde aathundu...god bless”   ·   “concept superb & comedy superb”   ·   "
             "“Movie name dada?”", 13, False, INK, "Georgia"),
         run("   →  all labelled Discouraging · Hopelessness", 12, False, MUTED)],
        [run("Not a row shift: comparing each prediction with the label 1–5 rows away "
             "agrees no better (0.31–0.39, vs 0.33 for the matching row).", 12, False, MUTED)],
    ], anchor="m", after=6)
    caption(s, 0.6, 6.95, 12.1, "This split was made after scoring, to explain the result; "
                                "the official test score is the full 744 comments.")
    notes(s, "Why is test so much lower than dev? The labelled test file was released as two "
             "merged parts; in the second part the label columns were swapped, which we "
             "repaired in phase 1. The first part looks like train and dev and M3 scores 0.410 "
             "on it. The second part has a very different label mix, almost no Blended Hope, "
             "and every model's tone accuracy there is about one in three, which is chance. "
             "Its raw labels include clearly positive comments marked Discouraging. We "
             "checked: the labels are not simply shifted by a few rows. This split is an "
             "explanation, not a second official score.")
    return s


def slide_takeaways(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "TAKEAWAYS", "What we learned")
    items = [
        ("The baseline is strong on dev",
         "M1 is best on tone (0.579), but it drops the most on test."),
        ("Pretraining holds up best",
         "MuRIL leads on hope type, and M3 multi-task is the best system on test (0.364)."),
        ("Multi-task gave no clear gain",
         "Two heads ≈ separate models; a pair head helps hope type (0.437) but not tone."),
        ("Diagnose before tuning",
         "MuRIL's collapse, M4's tiny pairs and the test drop were each found by looking inside."),
        ("Small dev gains are noise",
         "Combinations won on dev (+0.011) but lost on test (−0.023)."),
        ("Check the labels too",
         "One part of the test file scores at chance; its labels, not the models, look wrong."),
    ]
    for i, (title, body) in enumerate(items):
        col, row = i % 3, i // 3
        x, y = 0.6 + col * 4.1, 1.65 + row * 2.15
        add_box(s, x, y, 3.9, 1.95, CARD)
        badge(s, x + 0.22, y + 0.25, 0.6, str(i + 1), AMBER, INK, 18)
        add_text(s, x + 0.22, y + 0.98, 3.5, 0.9, [[run(title, 15, True, INK)],
                                                  [run(body, 12, False, MUTED)]], after=3)
    add_box(s, 0.6, 6.05, 12.1, 0.85, WHITE, line=TEAL)
    add_text(s, 0.85, 6.1, 11.6, 0.75, [[run("Why scores are modest overall:  ", 14, True, INK),
                                         run("Tulu is not among MuRIL's languages, comments are "
                                             "about 5 words long, the tone classes overlap, and "
                                             "part of the test labels look unreliable.", 14,
                                             False, INK)]], anchor="m")
    notes(s, "Six things we learned. A simple TF-IDF model is very strong on dev, but it does "
             "not hold up on test. Pretraining holds up best. Predicting both labels together "
             "did not clearly help, though one head over the pairs helps hope type. Looking "
             "inside a model beats guessing settings. Gains of one point on 753 dev comments "
             "are within noise. And when scores drop, check the labels as well as the model.")
    return s


def slide_limitations(prs, layout):
    s = prs.slides.add_slide(layout)
    header(s, "LIMITATIONS AND NEXT STEPS", "What limits these results, and what to try next")
    columns = [
        ("Limitations", RED, WHITE, "!", [
            ("Unreliable test labels", "rows 429–744 score at chance, so the official test "
                                       "score understates every model"),
            ("Not exactly repeatable", "GPU training differs slightly between runs even with "
                                       "fixed seeds; we report 5-seed means"),
            ("Small, short data", "3,365 comments of about 5 words; Tulu is not one of "
                                  "MuRIL's languages"),
            ("Ensembles cost more", "a combination runs several models, so it is not one "
                                    "forward pass"),
        ]),
        ("Next steps", TEAL, WHITE, "→", [
            ("Re-check the test labels", "with Tulu speakers, starting from the 52 clear "
                                         "conflicts we listed"),
            ("Other encoders", "compare XLM-RoBERTa and IndicBERT with MuRIL"),
            ("More Tulu text", "continue MuRIL's pretraining on unlabelled Tulu comments"),
            ("Script normalisation", "transliterate Kannada-script comments to Roman script"),
        ]),
    ]
    for c, (title, fill, tcol, symbol, entries) in enumerate(columns):
        x = 0.6 + c * 6.15
        add_box(s, x, 1.65, 5.95, 4.15, CARD if c else CARD_AMBER)
        badge(s, x + 0.25, 1.85, 0.6, symbol, fill, tcol, 18)
        add_text(s, x + 1.05, 1.85, 4.6, 0.6, [[run(title, 17, True, INK)]], anchor="m")
        paras = []
        for head, body in entries:
            paras.append([run(head, 14, True, INK)])
            paras.append([run(body, 12, False, MUTED)])
        box = add_text(s, x + 0.3, 2.65, 5.35, 3.0, paras)
        for k, p in enumerate(box.text_frame.paragraphs):
            p.space_after = Pt(2 if k % 2 == 0 else 12)
    add_box(s, 0.6, 6.05, 12.1, 0.85, WHITE, line=TEAL)
    add_text(s, 0.85, 6.1, 11.6, 0.75, [[run("Most useful next step:  ", 14, True, INK),
                                         run("have Tulu speakers re-check test rows 429–744, "
                                             "then score the same declared systems again.",
                                             14, False, INK)]], anchor="m")
    notes(s, "The biggest limitation is the test labels: about 300 of the 744 test comments "
             "have labels that look unreliable, so the official test number understates all "
             "models. Next, we would re-check those labels with Tulu speakers, compare other "
             "multilingual encoders, and continue pretraining on unlabelled Tulu text.")
    return s


# ----------------------------------------------------------------- main

def main():
    prs = Presentation(SRC)
    old = list(prs.slides)             # the 17 midsem slides, in their midsem order
    layout = old[0].slide_layout       # "Blank", used by every midsem slide

    update_title(old[0])
    update_overview(old[1])
    update_results_table(old[12])
    update_per_class(old[13])
    update_protocol(old[15])
    update_appendix(old[16])

    m4_idea = slide_m4_idea(prs, layout)
    m4_results = slide_m4_results(prs, layout)
    combos = slide_combinations(prs, layout)
    test = slide_test(prs, layout)
    parts = slide_test_parts(prs, layout)
    takeaways = slide_takeaways(prs, layout)
    limits = slide_limitations(prs, layout)

    order = (old[0:12] + [m4_idea, m4_results, old[12], old[13], old[15], combos, test, parts,
                          takeaways, limits, old[16]])

    # The midsem takeaways slide (old[14]) is replaced by the new one: drop it.
    # The slide order lives in presentation.xml as a list of slide ids (<p:sldIdLst>):
    # take every entry out, then put them back in the new order.
    id_list = prs.slides._sldIdLst
    by_slide = {}
    for sid in list(id_list):
        slide = prs.slides.part.related_part(sid.rId).slide
        by_slide[id(slide)] = sid
        id_list.remove(sid)
    for slide in order:
        id_list.append(by_slide[id(slide)])
    dropped = by_slide[id(old[14])]
    prs.part.drop_rel(dropped.rId)

    # Page numbers "n / N" on every slide except the title slide.
    total = len(order)
    for n, slide in enumerate(order, start=1):
        if n == 1:
            continue
        found = False
        for sh in slide.shapes:
            if sh.has_text_frame and " / " in sh.text_frame.text \
                    and sh.text_frame.text.replace(" / ", "").isdigit():
                sh.text_frame.paragraphs[0].runs[0].text = f"{n} / {total}"
                found = True
        if not found:
            add_text(slide, 12.0, 7.05, 0.8, 0.3, [[run(f"{n} / {total}", 10, False, MUTED)]],
                     align="r")

    prs.save(OUT)
    print(f"saved {OUT}  ({total} slides)")


if __name__ == "__main__":
    main()
