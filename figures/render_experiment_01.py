"""Render the Experiment 1 figure as a vector PDF and 300-dpi PNG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BLUE = (164, 30, 46)       # Pattern Recognition Letters crimson emphasis
ORANGE = (92, 92, 92)      # neutral comparison
DARK = (34, 34, 34)
GRID = (220, 220, 220)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def log_map(value, low, high, start, end):
    return start + (math.log10(value) - math.log10(low)) / (
        math.log10(high) - math.log10(low)
    ) * (end - start)


def render_png(data, path):
    scale = 3
    width, height = 2100, 870
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_path = Path(r"C:\Windows\Fonts\times.ttf")
    bold_path = Path(r"C:\Windows\Fonts\timesbd.ttf")
    regular = ImageFont.truetype(str(regular_path), 23 * scale // 2)
    small = ImageFont.truetype(str(regular_path), 19 * scale // 2)
    bold = ImageFont.truetype(str(bold_path), 24 * scale // 2)
    panel_w, gap, left = 840, 170, 140
    top, bottom = 130, 700

    def text_center(x, y, value, font=regular, fill=DARK):
        box = draw.textbbox((0, 0), value, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y), value, font=font, fill=fill)

    def vertical_text(x, y, value):
        box = draw.textbbox((0, 0), value, font=regular)
        label = Image.new(
            "RGBA", (box[2] - box[0] + 12, box[3] - box[1] + 12), (255, 255, 255, 0)
        )
        ImageDraw.Draw(label).text((6, 2), value, font=regular, fill=DARK)
        label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(label, (int(x - label.width / 2), int(y - label.height / 2)), label)

    def dashed_segment(p0, p1, fill, width=5, dash=18, gap=12):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            draw.line((x0 + ux * pos, y0 + uy * pos,
                       x0 + ux * end, y0 + uy * end), fill=fill, width=width)
            pos += dash + gap

    for panel in range(2):
        x0 = left + panel * (panel_w + gap)
        draw.line((x0, top, x0, bottom), fill=DARK, width=3)
        draw.line((x0, bottom, x0 + panel_w, bottom), fill=DARK, width=3)

    # Panel a: numerical equivalence.
    x0 = left
    text_center(x0 + panel_w / 2, 35, "Streaming additivity", bold)
    draw.text((x0 - 70, 35), "(a)", font=bold, fill=DARK)
    chunks = [entry["chunks"] for entry in data["equivalence"]["ordered"]]
    ymin, ymax = 1e-9, 1e-5
    for exponent in range(-9, -4):
        value = 10.0**exponent
        y = bottom - log_map(value, ymin, ymax, 0, bottom - top)
        draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=2)
        draw.text((x0 - 78, y - 14), f"1e{exponent}", font=small, fill=DARK)
    xs = [x0 + i * panel_w / (len(chunks) - 1) for i in range(len(chunks))]
    for x, chunk in zip(xs, chunks):
        draw.line((x, bottom, x, bottom + 8), fill=DARK, width=2)
        text_center(x, bottom + 14, str(chunk), small)
    for order, color in (("ordered", BLUE), ("shuffled", ORANGE)):
        values = [max(entry["max_relative_error"], ymin) for entry in data["equivalence"][order]]
        ys = [bottom - log_map(value, ymin, ymax, 0, bottom - top) for value in values]
        if order == "shuffled":
            for p0, p1 in zip(list(zip(xs, ys))[:-1], list(zip(xs, ys))[1:]):
                dashed_segment(p0, p1, color)
        else:
            draw.line(list(zip(xs, ys)), fill=color, width=5, joint="curve")
        for x, y in zip(xs, ys):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
    text_center(x0 + panel_w / 2, bottom + 62, "Number of streaming chunks", regular)
    draw.text((x0 + 20, top + 18), "maximum over 32 shapes", font=small, fill=DARK)
    draw.line((x0 + 570, top + 25, x0 + 625, top + 25), fill=BLUE, width=5)
    draw.text((x0 + 635, top + 9), "ordered", font=small, fill=DARK)
    dashed_segment((x0 + 570, top + 65), (x0 + 625, top + 65), ORANGE)
    draw.text((x0 + 635, top + 49), "shuffled", font=small, fill=DARK)
    vertical_text(24, (top + bottom) / 2, "Relative state error")

    # Panel b: memory scaling.
    x0 = left + panel_w + gap
    text_center(x0 + panel_w / 2, 35, "Retained-state memory", bold)
    draw.text((x0 - 70, 35), "(b)", font=bold, fill=DARK)
    xmin, xmax, mymin, mymax = 1e3, 3e5, 1e-2, 10.0
    for exponent in range(-2, 2):
        value = 10.0**exponent
        y = bottom - log_map(value, mymin, mymax, 0, bottom - top)
        draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=2)
        draw.text((x0 - 63, y - 14), f"{value:g}", font=small, fill=DARK)
    xticks = [1e3, 1e4, 1e5]
    for value in xticks:
        x = log_map(value, xmin, xmax, x0, x0 + panel_w)
        draw.line((x, bottom, x, bottom + 8), fill=DARK, width=2)
        text_center(x, bottom + 14, f"1e{int(math.log10(value))}", small)
    curve = data["memory_curve"]
    xvalues = [entry["cumulative_points"] for entry in curve]
    state = [entry["pointprogrammer_state_bytes"] / 2**20 for entry in curve]
    raw = [entry["raw_xyz_bytes"] / 2**20 for entry in curve]
    px = [log_map(value, xmin, xmax, x0, x0 + panel_w) for value in xvalues]
    for values, color in ((state, BLUE), (raw, ORANGE)):
        py = [bottom - log_map(value, mymin, mymax, 0, bottom - top) for value in values]
        draw.line(list(zip(px, py)), fill=color, width=5, joint="curve")
    text_center(x0 + panel_w / 2, bottom + 62, "Cumulative observed points", regular)
    vertical_text(x0 - 98, (top + bottom) / 2, "Memory (MiB)")
    draw.line((x0 + 500, top + 25, x0 + 555, top + 25), fill=BLUE, width=5)
    draw.text((x0 + 565, top + 9), "FWP state", font=small, fill=DARK)
    draw.line((x0 + 500, top + 65, x0 + 555, top + 65), fill=ORANGE, width=5)
    draw.text((x0 + 565, top + 49), "raw XYZ", font=small, fill=DARK)
    state_kib = data["state_kib_per_sample"]
    draw.text((x0 + 22, top + 20), f"fixed at {state_kib:.1f} KiB", font=small, fill=BLUE)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300))


def render_pdf(data, path):
    page_w, page_h = 7.2 * 72, 3.05 * 72
    c = canvas.Canvas(str(path), pagesize=(page_w, page_h))
    regular_path = Path(r"C:\Windows\Fonts\times.ttf")
    bold_path = Path(r"C:\Windows\Fonts\timesbd.ttf")
    if regular_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont("FigureTimes", str(regular_path)))
        pdfmetrics.registerFont(TTFont("FigureTimesBold", str(bold_path)))
        regular, bold = "FigureTimes", "FigureTimesBold"
    else:
        regular, bold = "Times-Roman", "Times-Bold"
    blue = tuple(v / 255 for v in BLUE)
    orange = tuple(v / 255 for v in ORANGE)
    dark = tuple(v / 255 for v in DARK)
    grid = tuple(v / 255 for v in GRID)
    margin, gap, top, bottom = 39, 42, 176, 39
    panel_w = (page_w - 2 * margin - gap) / 2

    def centered(x, y, value, size=8, font=regular):
        c.setFont(font, size)
        c.setFillColorRGB(*dark)
        c.drawCentredString(x, y, value)

    for panel in range(2):
        x0 = margin + panel * (panel_w + gap)
        c.setStrokeColorRGB(*dark); c.setLineWidth(0.8)
        c.line(x0, bottom, x0, top); c.line(x0, bottom, x0 + panel_w, bottom)

    x0 = margin
    centered(x0 + panel_w / 2, 202, "Streaming additivity", 9, bold)
    centered(x0 - 15, 202, "(a)", 9, bold)
    ymin, ymax = 1e-9, 1e-5
    for exponent in range(-9, -4):
        value = 10.0**exponent
        y = bottom + log_map(value, ymin, ymax, 0, top - bottom)
        c.setStrokeColorRGB(*grid); c.setLineWidth(0.35); c.line(x0, y, x0 + panel_w, y)
        c.setFont(regular, 6.8); c.setFillColorRGB(*dark); c.drawRightString(x0 - 4, y - 2, f"1e{exponent}")
    chunks = [entry["chunks"] for entry in data["equivalence"]["ordered"]]
    xs = [x0 + i * panel_w / (len(chunks) - 1) for i in range(len(chunks))]
    for x, value in zip(xs, chunks): centered(x, bottom - 11, str(value), 7)
    for order, color in (("ordered", blue), ("shuffled", orange)):
        values = [max(entry["max_relative_error"], ymin) for entry in data["equivalence"][order]]
        ys = [bottom + log_map(value, ymin, ymax, 0, top - bottom) for value in values]
        c.setStrokeColorRGB(*color); c.setFillColorRGB(*color); c.setLineWidth(1.5)
        c.setDash(4, 2.5) if order == "shuffled" else c.setDash()
        path_obj = c.beginPath(); path_obj.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]): path_obj.lineTo(x, y)
        c.drawPath(path_obj)
        c.setDash()
        for x, y in zip(xs, ys): c.circle(x, y, 1.8, fill=1, stroke=0)
    centered(x0 + panel_w / 2, 10, "Number of streaming chunks", 8)
    c.saveState(); c.translate(9, (top + bottom) / 2); c.rotate(90); centered(0, 0, "Relative state error", 8); c.restoreState()
    c.setFont(regular, 6.5); c.setFillColorRGB(*dark); c.drawString(x0 + 5, top - 10, "maximum over 32 shapes")
    c.setStrokeColorRGB(*blue); c.setLineWidth(1.5); c.line(x0 + panel_w - 73, top - 8, x0 + panel_w - 58, top - 8)
    c.setFont(regular, 6.5); c.setFillColorRGB(*dark); c.drawString(x0 + panel_w - 54, top - 10, "ordered")
    c.setStrokeColorRGB(*orange); c.setDash(4, 2.5); c.line(x0 + panel_w - 73, top - 19, x0 + panel_w - 58, top - 19); c.setDash()
    c.setFillColorRGB(*dark); c.drawString(x0 + panel_w - 54, top - 21, "shuffled")

    x0 = margin + panel_w + gap
    centered(x0 + panel_w / 2, 202, "Retained-state memory", 9, bold)
    centered(x0 - 15, 202, "(b)", 9, bold)
    xmin, xmax, mymin, mymax = 1e3, 3e5, 1e-2, 10.0
    for exponent in range(-2, 2):
        value = 10.0**exponent
        y = bottom + log_map(value, mymin, mymax, 0, top - bottom)
        c.setStrokeColorRGB(*grid); c.setLineWidth(0.35); c.line(x0, y, x0 + panel_w, y)
        c.setFont(regular, 6.8); c.setFillColorRGB(*dark); c.drawRightString(x0 - 4, y - 2, f"{value:g}")
    for value in [1e3, 1e4, 1e5]:
        x = log_map(value, xmin, xmax, x0, x0 + panel_w)
        centered(x, bottom - 11, f"1e{int(math.log10(value))}", 7)
    curve = data["memory_curve"]
    xvalues = [entry["cumulative_points"] for entry in curve]
    px = [log_map(value, xmin, xmax, x0, x0 + panel_w) for value in xvalues]
    for key, color in (("pointprogrammer_state_bytes", blue), ("raw_xyz_bytes", orange)):
        values = [entry[key] / 2**20 for entry in curve]
        ys = [bottom + log_map(value, mymin, mymax, 0, top - bottom) for value in values]
        c.setStrokeColorRGB(*color); c.setLineWidth(1.5)
        path_obj = c.beginPath(); path_obj.moveTo(px[0], ys[0])
        for x, y in zip(px[1:], ys[1:]): path_obj.lineTo(x, y)
        c.drawPath(path_obj)
    centered(x0 + panel_w / 2, 10, "Cumulative observed points", 8)
    c.saveState(); c.translate(x0 - 29, (top + bottom) / 2); c.rotate(90); centered(0, 0, "Memory (MiB)", 8); c.restoreState()
    c.setFont(regular, 6.5); c.setFillColorRGB(*blue); c.drawString(x0 + 5, top - 10, f"fixed at {data['state_kib_per_sample']:.1f} KiB")
    c.setStrokeColorRGB(*blue); c.setLineWidth(1.5); c.line(x0 + panel_w - 92, top - 8, x0 + panel_w - 77, top - 8)
    c.setFont(regular, 6.5); c.setFillColorRGB(*dark); c.drawString(x0 + panel_w - 73, top - 10, "FWP state")
    c.setStrokeColorRGB(*orange); c.line(x0 + panel_w - 92, top - 19, x0 + panel_w - 77, top - 19)
    c.setFillColorRGB(*dark); c.drawString(x0 + panel_w - 73, top - 21, "raw XYZ")
    c.showPage(); c.save()


def main():
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    render_png(data, args.output_stem.with_suffix(".png"))
    render_pdf(data, args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
