"""Render Experiment 2 as PDF and PNG without plotting dependencies."""

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
ORANGE = (70, 70, 70)
GREEN = (125, 125, 125)
PURPLE = (180, 180, 180)
DARK = (34, 34, 34)
GRID = (220, 220, 220)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def log_position(value, low, high, start, end):
    return start + (math.log10(value) - math.log10(low)) / (
        math.log10(high) - math.log10(low)
    ) * (end - start)


def render_png(data, path):
    width, height = 2100, 870
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 29)
    small = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 25)
    bold = ImageFont.truetype(r"C:\Windows\Fonts\timesbd.ttf", 34)
    left, panel_w, gap, top, bottom = 140, 840, 170, 130, 700

    def center(x, y, value, font=regular, fill=DARK):
        box = draw.textbbox((0, 0), value, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y), value, font=font, fill=fill)

    def vertical(x, y, value):
        box = draw.textbbox((0, 0), value, font=regular)
        label = Image.new("RGBA", (box[2] + 12, box[3] + 12), (255, 255, 255, 0))
        ImageDraw.Draw(label).text((6, 2), value, font=regular, fill=DARK)
        label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(label, (int(x - label.width / 2), int(y - label.height / 2)), label)

    for panel in range(2):
        x0 = left + panel * (panel_w + gap)
        draw.line((x0, top, x0, bottom), fill=DARK, width=3)
        draw.line((x0, bottom, x0 + panel_w, bottom), fill=DARK, width=3)

    x0 = left
    center(x0 + panel_w / 2, 35, "Streaming state operations", bold)
    draw.text((x0 - 70, 35), "(a)", font=bold, fill=DARK)
    ymin, ymax = 1e-9, 1e-5
    for exponent in range(-9, -4):
        value = 10.0**exponent
        y = bottom - log_position(value, ymin, ymax, 0, bottom - top)
        draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=2)
        draw.text((x0 - 78, y - 14), f"1e{exponent}", font=small, fill=DARK)
    operations = ["append", "merge", "remove", "decay"]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    spacing = panel_w / len(operations)
    bar_w = 95
    for index, (name, color) in enumerate(zip(operations, colors)):
        value = max(data["operations"][name]["max_relative_error"], ymin)
        y = bottom - log_position(value, ymin, ymax, 0, bottom - top)
        x = x0 + spacing * (index + 0.5)
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
        center(x, bottom + 14, name, small)
    vertical(24, (top + bottom) / 2, "Maximum relative error")
    center(x0 + panel_w / 2, bottom + 62, "State operation", regular)

    x0 = left + panel_w + gap
    center(x0 + panel_w / 2, 35, "Bounded decayed state", bold)
    draw.text((x0 - 70, 35), "(b)", font=bold, fill=DARK)
    curve = data["state_norm_curve"]
    xmax = curve[-1]["frame"]
    ymax = math.ceil(max(item["additive"] for item in curve) * 1.08)
    for value in range(0, ymax + 1, 4):
        y = bottom - value / ymax * (bottom - top)
        draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=2)
        draw.text((x0 - 48, y - 14), str(value), font=small, fill=DARK)
    xs = [x0 + (item["frame"] - 1) / (xmax - 1) * panel_w for item in curve]
    for frame in [1, 4, 8, 12, 16]:
        x = x0 + (frame - 1) / (xmax - 1) * panel_w
        center(x, bottom + 14, str(frame), small)
    for key, color in (("additive", ORANGE), ("decay", BLUE)):
        ys = [bottom - item[key] / ymax * (bottom - top) for item in curve]
        draw.line(list(zip(xs, ys)), fill=color, width=5, joint="curve")
    center(x0 + panel_w / 2, bottom + 62, "Observed frames", regular)
    vertical(x0 - 98, (top + bottom) / 2, "State norm / first-frame norm")
    draw.line((x0 + 535, top + 25, x0 + 590, top + 25), fill=ORANGE, width=5)
    draw.text((x0 + 600, top + 9), "additive", font=small, fill=DARK)
    draw.line((x0 + 535, top + 65, x0 + 590, top + 65), fill=BLUE, width=5)
    draw.text((x0 + 600, top + 49), f"decay, rho={data['decay']:.2f}", font=small, fill=DARK)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300))


def render_pdf(data, path):
    w, h = 7.2 * 72, 3.05 * 72
    c = canvas.Canvas(str(path), pagesize=(w, h))
    pdfmetrics.registerFont(TTFont("FigureTimes", r"C:\Windows\Fonts\times.ttf"))
    pdfmetrics.registerFont(TTFont("FigureTimesBold", r"C:\Windows\Fonts\timesbd.ttf"))
    regular, bold = "FigureTimes", "FigureTimesBold"
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    rgb = lambda color: tuple(v / 255 for v in color)
    dark, grid = rgb(DARK), rgb(GRID)
    margin, gap, top, bottom = 39, 42, 176, 39
    panel_w = (w - 2 * margin - gap) / 2

    def center(x, y, value, size=8, font=regular):
        c.setFont(font, size); c.setFillColorRGB(*dark); c.drawCentredString(x, y, value)

    for panel in range(2):
        x0 = margin + panel * (panel_w + gap)
        c.setStrokeColorRGB(*dark); c.setLineWidth(0.8)
        c.line(x0, bottom, x0, top); c.line(x0, bottom, x0 + panel_w, bottom)

    x0 = margin
    center(x0 + panel_w / 2, 202, "Streaming state operations", 9, bold)
    center(x0 - 15, 202, "(a)", 9, bold)
    ymin, ymax = 1e-9, 1e-5
    for exponent in range(-9, -4):
        value = 10.0**exponent
        y = bottom + log_position(value, ymin, ymax, 0, top - bottom)
        c.setStrokeColorRGB(*grid); c.setLineWidth(0.35); c.line(x0, y, x0 + panel_w, y)
        c.setFont(regular, 6.8); c.setFillColorRGB(*dark); c.drawRightString(x0 - 4, y - 2, f"1e{exponent}")
    operations = ["append", "merge", "remove", "decay"]
    spacing = panel_w / len(operations)
    for index, (name, color) in enumerate(zip(operations, colors)):
        value = max(data["operations"][name]["max_relative_error"], ymin)
        y = bottom + log_position(value, ymin, ymax, 0, top - bottom)
        x = x0 + spacing * (index + 0.5)
        c.setFillColorRGB(*rgb(color)); c.rect(x - 11, bottom, 22, y - bottom, fill=1, stroke=0)
        center(x, bottom - 11, name, 7)
    center(x0 + panel_w / 2, 10, "State operation", 8)
    c.saveState(); c.translate(9, (top + bottom) / 2); c.rotate(90); center(0, 0, "Maximum relative error", 8); c.restoreState()

    x0 = margin + panel_w + gap
    center(x0 + panel_w / 2, 202, "Bounded decayed state", 9, bold)
    center(x0 - 15, 202, "(b)", 9, bold)
    curve = data["state_norm_curve"]
    xmax = curve[-1]["frame"]
    ymax = math.ceil(max(item["additive"] for item in curve) * 1.08)
    for value in range(0, ymax + 1, 4):
        y = bottom + value / ymax * (top - bottom)
        c.setStrokeColorRGB(*grid); c.setLineWidth(0.35); c.line(x0, y, x0 + panel_w, y)
        c.setFont(regular, 6.8); c.setFillColorRGB(*dark); c.drawRightString(x0 - 4, y - 2, str(value))
    xs = [x0 + (item["frame"] - 1) / (xmax - 1) * panel_w for item in curve]
    for frame in [1, 4, 8, 12, 16]:
        x = x0 + (frame - 1) / (xmax - 1) * panel_w
        center(x, bottom - 11, str(frame), 7)
    for key, color in (("additive", ORANGE), ("decay", BLUE)):
        ys = [bottom + item[key] / ymax * (top - bottom) for item in curve]
        c.setStrokeColorRGB(*rgb(color)); c.setLineWidth(1.5)
        p = c.beginPath(); p.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]): p.lineTo(x, y)
        c.drawPath(p)
    center(x0 + panel_w / 2, 10, "Observed frames", 8)
    c.saveState(); c.translate(x0 - 29, (top + bottom) / 2); c.rotate(90); center(0, 0, "State norm / first-frame norm", 8); c.restoreState()
    c.setStrokeColorRGB(*rgb(ORANGE)); c.line(x0 + panel_w - 88, top - 8, x0 + panel_w - 73, top - 8)
    c.setFont(regular, 6.5); c.setFillColorRGB(*dark); c.drawString(x0 + panel_w - 69, top - 10, "additive")
    c.setStrokeColorRGB(*rgb(BLUE)); c.line(x0 + panel_w - 88, top - 19, x0 + panel_w - 73, top - 19)
    c.setFillColorRGB(*dark); c.drawString(x0 + panel_w - 69, top - 21, f"decay, rho={data['decay']:.2f}")
    c.showPage(); c.save()


def main():
    parsed = args()
    data = json.loads(parsed.input.read_text(encoding="utf-8"))
    # Accept historical result files while presenting the operation accurately.
    if "decay" not in data["operations"] and "delta" in data["operations"]:
        data["operations"]["decay"] = data["operations"]["delta"]
    for item in data["state_norm_curve"]:
        if "decay" not in item and "delta" in item:
            item["decay"] = item["delta"]
    render_png(data, parsed.output_stem.with_suffix(".png"))
    render_pdf(data, parsed.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
