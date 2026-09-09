"""Render the PointProgrammerNet system architecture as a vector PDF."""

from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
OUT = HERE / "fig00_pointprogrammernet_architecture.pdf"

PAGE_W = 7.2 * 72
PAGE_H = 6.15 * 72

DARK = (34 / 255, 34 / 255, 34 / 255)
MUTED = (96 / 255, 96 / 255, 96 / 255)
BLUE = (164 / 255, 30 / 255, 46 / 255)
BLUE_FILL = (247 / 255, 247 / 255, 247 / 255)
PURPLE = (164 / 255, 30 / 255, 46 / 255)
PURPLE_FILL = (253 / 255, 245 / 255, 246 / 255)
GREEN = (58 / 255, 58 / 255, 58 / 255)
GREEN_FILL = (1, 1, 1)
ORANGE = (58 / 255, 58 / 255, 58 / 255)
ORANGE_FILL = (1, 1, 1)
GRAY_FILL = (246 / 255, 246 / 255, 246 / 255)


def register_fonts():
    regular_path = Path(r"C:\Windows\Fonts\times.ttf")
    bold_path = Path(r"C:\Windows\Fonts\timesbd.ttf")
    if regular_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont("ArchTimes", str(regular_path)))
        pdfmetrics.registerFont(TTFont("ArchTimesBold", str(bold_path)))
        return "ArchTimes", "ArchTimesBold"
    return "Times-Roman", "Times-Bold"


REGULAR, BOLD = register_fonts()


def draw_text(c, x, y, text, size=7.4, font=REGULAR, color=DARK, align="center"):
    c.setFont(font, size)
    c.setFillColorRGB(*color)
    if align == "left":
        c.drawString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawCentredString(x, y, text)


def box(c, x, y, w, h, lines, fill, stroke, title=None, size=7.1, radius=5):
    c.setFillColorRGB(*fill)
    c.setStrokeColorRGB(*stroke)
    c.setLineWidth(0.9)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    if title:
        draw_text(c, x + w / 2, y + h - 12, title, 8.0, BOLD, stroke)
        line_y = y + h - 18
        c.setStrokeColorRGB(*stroke)
        c.setLineWidth(0.35)
        c.line(x + 8, line_y, x + w - 8, line_y)
        available_top = line_y - 9
    else:
        available_top = y + h - 12
    gap = 9.2
    total = (len(lines) - 1) * gap
    first = available_top - max(0, (available_top - (y + 8) - total) / 2)
    for i, line in enumerate(lines):
        draw_text(c, x + w / 2, first - i * gap, line, size, REGULAR, DARK)


def panel(c, x, y, w, h, title, subtitle, color, fill):
    c.setFillColorRGB(*fill)
    c.setStrokeColorRGB(*color)
    c.setLineWidth(1.15)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
    draw_text(c, x + 12, y + h - 15, title, 9.4, BOLD, color, "left")
    draw_text(c, x + w - 12, y + h - 14, subtitle, 6.7, REGULAR, MUTED, "right")


def arrow(c, points, color=DARK, width=0.9, head=4.2):
    """Draw an orthogonal polyline and an arrowhead on its final segment."""
    c.setStrokeColorRGB(*color)
    c.setFillColorRGB(*color)
    c.setLineWidth(width)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 >= x0 else -1
        tip = (x1, y1)
        base = x1 - sign * head
        tri = [(tip[0], tip[1]), (base, y1 + head * 0.62), (base, y1 - head * 0.62)]
    else:
        sign = 1 if y1 >= y0 else -1
        tip = (x1, y1)
        base = y1 - sign * head
        tri = [(tip[0], tip[1]), (x1 + head * 0.62, base), (x1 - head * 0.62, base)]
    p = c.beginPath()
    p.moveTo(*tri[0])
    p.lineTo(*tri[1])
    p.lineTo(*tri[2])
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def render():
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("PointProgrammerNet system architecture")
    margin = 12

    # Shared encoder.
    enc_y, enc_h = 296, 135
    panel(c, margin, enc_y, PAGE_W - 2 * margin, enc_h,
          "Shared additive FWP encoder", "", BLUE, BLUE_FILL)
    draw_text(c, 278, 414, "unordered, mergeable, fixed retained state",
              6.7, REGULAR, MUTED)

    box(c, 24, 342, 76, 45, ["point or", "stream chunk", "{x_i}"], GRAY_FILL, BLUE, size=7.2)
    box(c, 119, 326, 125, 77,
        ["continuous key k_s(x_i)", "branch MLP g_s(x_i)",
         "second-order value v_s(x_i)"], GRAY_FILL, BLUE,
        title="Per-point program", size=7.0)
    box(c, 263, 342, 83, 45, ["sum outer products", "S_s += k_s v_s^T"], GRAY_FILL, BLUE, size=6.9)

    sx, sw, sh = 368, 125, 27
    state_ys = [390, 356, 322]
    state_names = [
        "fine  S_f   128 x 170   r0=0.08",
        "middle S_m  128 x 170   r0=0.20",
        "broad S_b   128 x 170   r0=0.60",
    ]
    for sy, label in zip(state_ys, state_names):
        box(c, sx, sy, sw, sh, [label], PURPLE_FILL, PURPLE, size=6.8, radius=4)

    arrow(c, [(100, 364.5), (119, 364.5)], BLUE)
    arrow(c, [(244, 364.5), (263, 364.5)], BLUE)
    # Orthogonal fan-out from the write to all three states.
    c.setStrokeColorRGB(*BLUE); c.setLineWidth(0.9)
    c.line(346, 364.5, 357, 364.5)
    c.line(357, 335.5, 357, 403.5)
    for sy in [403.5, 369.5, 335.5]:
        arrow(c, [(357, sy), (368, sy)], BLUE)

    draw_text(c, 181.5, 311, "same operation for s in {fine, middle, broad}", 6.7, REGULAR, MUTED)
    draw_text(c, 430.5, 307, "3 x 128 x 170 float32 = 255 KiB", 6.8, BOLD, PURPLE)
    draw_text(c, 430.5, 298, "historical points may now be discarded", 6.6, REGULAR, PURPLE)

    # Join all three states into one fixed-state bus, then feed both heads.
    bus_y = 286
    seg_drop_x = 157.5
    cls_drop_x = 331.5
    c.setStrokeColorRGB(*PURPLE); c.setLineWidth(1.0)
    c.line(502, 335.5, 502, 403.5)
    for sy in [403.5, 369.5, 335.5]:
        c.line(493, sy, 502, sy)
    c.line(502, bus_y, 502, 335.5)
    # Leave a deliberate gap for the label so no rule crosses the text.
    c.line(seg_drop_x, bus_y, 216, bus_y)
    c.line(306, bus_y, 502, bus_y)
    c.line(seg_drop_x, bus_y, seg_drop_x, 273)
    c.line(cls_drop_x, bus_y, cls_drop_x, 273)
    draw_text(c, 261, bus_y - 2.2, "fixed states only", 6.5, BOLD, PURPLE)

    # Segmentation panel.
    px, py, pw, ph = 12, 18, 243, 255
    panel(c, px, py, pw, ph, "(a) Segmentation head", "coordinate-conditioned field", GREEN, GREEN_FILL)
    box(c, 112, 213, 91, 36, ["three state reads", "a_s(q)=k_s(q)^T S_s"], GRAY_FILL, GREEN, size=6.7)
    box(c, 112, 158, 91, 35, ["moment decode", "+ branch MLPs"], GRAY_FILL, GREEN, size=6.8)
    box(c, 25, 213, 67, 36, ["query coordinate", "q"], GRAY_FILL, GREEN, size=7.0)
    box(c, 25, 158, 67, 35, ["coordinate skip", "phi(q)"], GRAY_FILL, GREEN, size=6.9)
    box(c, 25, 107, 67, 31, ["category code", "e_cat"], GRAY_FILL, GREEN, size=6.8)
    box(c, 120, 105, 76, 40, ["concatenate", "[r_m;r_f;r_b;", "phi(q);e_cat]"], GRAY_FILL, GREEN, size=6.5)
    box(c, 112, 57, 91, 30, ["pointwise MLP"], GRAY_FILL, GREEN, size=7.0)
    box(c, 112, 25, 91, 23, ["part logits at q"], GREEN_FILL, GREEN, size=6.8)

    arrow(c, [(seg_drop_x, 273), (seg_drop_x, 249)], PURPLE)
    arrow(c, [(92, 231), (112, 231)], GREEN)
    arrow(c, [(157.5, 213), (157.5, 193)], GREEN)
    arrow(c, [(58.5, 213), (58.5, 193)], GREEN)
    arrow(c, [(157.5, 158), (157.5, 145)], GREEN)
    # Independent ports make the U-Net-like coordinate skip and category code
    # unambiguous: they never share an edge before concatenation.
    arrow(c, [(92, 175.5), (106, 175.5), (106, 134), (120, 134)], GREEN)
    arrow(c, [(92, 122.5), (110, 122.5), (110, 116), (120, 116)], GREEN)
    arrow(c, [(158, 105), (158, 87)], GREEN)
    arrow(c, [(158, 57), (158, 48)], GREEN)
    draw_text(c, 29, 41, "independent queries", 6.3, BOLD, GREEN, "left")
    draw_text(c, 29, 31, "unseen q allowed", 6.3, REGULAR, GREEN, "left")
    draw_text(c, 29, 21, "no query interaction or max", 6.3, REGULAR, GREEN, "left")

    # Classification panel.
    px, py, pw, ph = 264, 18, PAGE_W - 276, 255
    panel(c, px, py, pw, ph, "(b) Classification head", "state-only decision", ORANGE, ORANGE_FILL)
    box(c, 285, 213, 93, 36, ["decode fixed rows", "coordinate-free stats"], GRAY_FILL, ORANGE, size=6.7)
    box(c, 399, 213, 92, 36, ["token MLP + two", "soft row summaries"], GRAY_FILL, ORANGE, size=6.7)
    box(c, 285, 153, 93, 39, ["main summary", "h_m"], GRAY_FILL, ORANGE, size=7.0)
    box(c, 399, 153, 92, 39, ["fine/broad summaries", "[h_f; h_b]"], GRAY_FILL, ORANGE, size=6.7)
    box(c, 399, 101, 92, 32, ["bounded residual", "delta"], GRAY_FILL, ORANGE, size=6.8)
    box(c, 310, 101, 56, 32, ["h_m + delta"], GRAY_FILL, ORANGE, size=6.9)
    box(c, 310, 57, 56, 27, ["LN + MLP"], GRAY_FILL, ORANGE, size=6.8)
    box(c, 300, 25, 76, 23, ["class logits"], ORANGE_FILL, ORANGE, size=6.9)

    arrow(c, [(cls_drop_x, 273), (cls_drop_x, 249)], PURPLE)
    arrow(c, [(378, 231), (399, 231)], ORANGE)
    c.setStrokeColorRGB(*ORANGE); c.setLineWidth(0.9)
    c.line(445, 213, 445, 202)
    c.line(331.5, 202, 445, 202)
    arrow(c, [(331.5, 202), (331.5, 192)], ORANGE)
    arrow(c, [(445, 202), (445, 192)], ORANGE)
    arrow(c, [(445, 153), (445, 133)], ORANGE)
    arrow(c, [(399, 117), (366, 117)], ORANGE)
    arrow(c, [(331.5, 153), (331.5, 133)], ORANGE)
    arrow(c, [(338, 101), (338, 84)], ORANGE)
    arrow(c, [(338, 57), (338, 48)], ORANGE)
    draw_text(c, 393, 74, "stored points = 0", 6.3, BOLD, ORANGE, "left")
    draw_text(c, 393, 62, "no terminal/global max", 6.3, REGULAR, ORANGE, "left")
    draw_text(c, 393, 50, "middle-radius LR: 25x", 6.3, REGULAR, ORANGE, "left")

    # Shared design statement.
    draw_text(c, PAGE_W / 2, 4.5,
              "Both heads consume the same fixed additive states; no decoded point set is reconstructed.",
              6.6, BOLD, DARK)
    c.showPage()
    c.save()


if __name__ == "__main__":
    render()
    print(OUT)
