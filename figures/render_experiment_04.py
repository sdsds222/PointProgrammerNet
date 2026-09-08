"""Render the streaming efficiency benchmark as a vector PDF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BLUE = (164 / 255, 30 / 255, 46 / 255)
ORANGE = (92 / 255, 92 / 255, 92 / 255)
DARK = (34 / 255, 34 / 255, 34 / 255)
GRID = (220 / 255, 220 / 255, 220 / 255)


def logmap(value, low, high, start, end):
    return start + (math.log2(value) - math.log2(low)) / (
        math.log2(high) - math.log2(low)
    ) * (end - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["timings"]
    width, height = 7.2 * 72, 3.05 * 72
    c = canvas.Canvas(str(args.output), pagesize=(width, height))
    pdfmetrics.registerFont(TTFont("FigureTimes", r"C:\Windows\Fonts\times.ttf"))
    pdfmetrics.registerFont(TTFont("FigureTimesBold", r"C:\Windows\Fonts\timesbd.ttf"))
    regular, bold = "FigureTimes", "FigureTimesBold"
    margin, gap, top, bottom = 42, 46, 176, 39
    panel_w = (width - 2 * margin - gap) / 2
    points = [row["cumulative_points"] for row in rows]
    xmin, xmax = min(points), max(points)

    def center(x, y, value, size=8, font=regular):
        c.setFont(font, size); c.setFillColorRGB(*DARK); c.drawCentredString(x, y, value)

    def axes(x0):
        c.setStrokeColorRGB(*DARK); c.setLineWidth(0.8)
        c.line(x0, bottom, x0, top); c.line(x0, bottom, x0 + panel_w, bottom)

    # Latency panel.
    x0 = margin; axes(x0)
    center(x0 + panel_w / 2, 202, "Incremental update latency", 9, bold)
    center(x0 - 15, 202, "(a)", 9, bold)
    latency_values = [
        row[key] for row in rows
        for key in ("incremental_ms_per_batch", "rebuild_ms_per_batch")
    ]
    ymin = 2 ** math.floor(math.log2(min(latency_values) * 0.75))
    ymax = 2 ** math.ceil(math.log2(max(latency_values) * 1.25))
    exponent = math.floor(math.log2(ymin))
    while 2**exponent <= ymax:
        value = 2**exponent
        y = logmap(value, ymin, ymax, bottom, top)
        c.setStrokeColorRGB(*GRID); c.setLineWidth(0.35); c.line(x0, y, x0 + panel_w, y)
        c.setFont(regular, 6.5); c.setFillColorRGB(*DARK); c.drawRightString(x0 - 4, y - 2, f"{value:g}")
        exponent += 1
    xs = [logmap(value, xmin, xmax, x0, x0 + panel_w) for value in points]
    for x, value in zip(xs, points):
        center(x, bottom - 11, str(value), 6.2)
    for key, trial_key, color in (
        ("incremental_ms_per_batch", "incremental_trials_ms", BLUE),
        ("rebuild_ms_per_batch", "rebuild_trials_ms", ORANGE),
    ):
        ys = [logmap(row[key], ymin, ymax, bottom, top) for row in rows]
        c.setStrokeColorRGB(*color); c.setFillColorRGB(*color); c.setLineWidth(1.5)
        for x, row in zip(xs, rows):
            low = logmap(min(row[trial_key]), ymin, ymax, bottom, top)
            high = logmap(max(row[trial_key]), ymin, ymax, bottom, top)
            c.line(x, low, x, high)
            c.line(x - 2, low, x + 2, low)
            c.line(x - 2, high, x + 2, high)
        p = c.beginPath(); p.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]): p.lineTo(x, y)
        c.drawPath(p)
        for x, y in zip(xs, ys): c.circle(x, y, 1.8, fill=1, stroke=0)
    center(x0 + panel_w / 2, 10, "Cumulative observed points", 8)
    c.saveState(); c.translate(10, (top + bottom) / 2); c.rotate(90); center(0, 0, "Latency per batch (ms)", 8); c.restoreState()
    c.setStrokeColorRGB(*BLUE); c.line(x0 + panel_w - 87, top - 8, x0 + panel_w - 72, top - 8)
    c.setFont(regular, 6.5); c.setFillColorRGB(*DARK); c.drawString(x0 + panel_w - 68, top - 10, "incremental")
    c.setStrokeColorRGB(*ORANGE); c.line(x0 + panel_w - 87, top - 19, x0 + panel_w - 72, top - 19)
    c.setFillColorRGB(*DARK); c.drawString(x0 + panel_w - 68, top - 21, "rebuild history")

    # Speedup panel.
    x0 = margin + panel_w + gap; axes(x0)
    center(x0 + panel_w / 2, 202, "Streaming speedup", 9, bold)
    center(x0 - 15, 202, "(b)", 9, bold)
    speedups = [row["speedup"] for row in rows]
    symax = max(2, math.ceil(max(speedups) * 1.15))
    tick_step = max(1, math.ceil(symax / 5))
    for value in range(0, symax + 1, tick_step):
        y = bottom + value / symax * (top - bottom)
        c.setStrokeColorRGB(*GRID); c.setLineWidth(0.35); c.line(x0, y, x0 + panel_w, y)
        c.setFont(regular, 6.5); c.setFillColorRGB(*DARK); c.drawRightString(x0 - 4, y - 2, str(value))
    xs = [logmap(value, xmin, xmax, x0, x0 + panel_w) for value in points]
    ys = [bottom + value / symax * (top - bottom) for value in speedups]
    c.setStrokeColorRGB(*BLUE); c.setFillColorRGB(*BLUE); c.setLineWidth(1.5)
    p = c.beginPath(); p.moveTo(xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:]): p.lineTo(x, y)
    c.drawPath(p)
    for x, y in zip(xs, ys): c.circle(x, y, 1.8, fill=1, stroke=0)
    for x, value in zip(xs, points): center(x, bottom - 11, str(value), 6.2)
    center(x0 + panel_w / 2, 10, "Cumulative observed points", 8)
    c.saveState(); c.translate(x0 - 29, (top + bottom) / 2); c.rotate(90); center(0, 0, "Rebuild / incremental speedup", 8); c.restoreState()
    max_error = max(row["max_relative_state_error"] for row in rows)
    c.setFont(regular, 6.5); c.setFillColorRGB(*DARK)
    c.drawString(x0 + 5, top - 10, f"state error <= {max_error:.1e}")
    c.showPage(); c.save()


if __name__ == "__main__":
    main()
