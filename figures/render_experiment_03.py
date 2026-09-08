"""Render the two-task frozen-architecture audit as a vector PDF."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    width, height = 7.2 * 72, 3.25 * 72
    pdf = canvas.Canvas(str(args.output), pagesize=(width, height))
    pdfmetrics.registerFont(TTFont("FigureTimes", r"C:\Windows\Fonts\times.ttf"))
    pdfmetrics.registerFont(TTFont("FigureTimesBold", r"C:\Windows\Fonts\timesbd.ttf"))
    regular, bold = "FigureTimes", "FigureTimesBold"
    margin, gap, bottom, top = 44, 42, 42, height - 43
    panel_width = (width - 2 * margin - gap) / 2

    for panel_index, task in enumerate(("Segmentation", "Classification")):
        rows = [row for row in data["comparisons"] if row["task"] == task]
        x0 = margin + panel_index * (panel_width + gap)
        all_changes = [
            metric["mean_change"] + metric["sd_change"]
            for row in rows for metric in row["metrics"]
        ]
        xmax = max(1.0, math.ceil(max(all_changes) * 2.0) / 2.0)

        pdf.setFont(bold, 9)
        pdf.setFillColorRGB(*DARK)
        title = "Segmentation write/state design" if panel_index == 0 else "Classification state readout"
        pdf.drawCentredString(x0 + panel_width / 2, height - 22, title)
        pdf.drawString(x0 - 14, height - 22, "(a)" if panel_index == 0 else "(b)")

        for tick_index in range(6):
            value = xmax * tick_index / 5
            x = x0 + value / xmax * panel_width
            pdf.setStrokeColorRGB(*GRID)
            pdf.setLineWidth(0.35)
            pdf.line(x, bottom, x, top)
            pdf.setFont(regular, 6.5)
            pdf.setFillColorRGB(*DARK)
            pdf.drawCentredString(x, bottom - 12, f"{value:.1f}")

        band = (top - bottom) / len(rows)
        for row_index, row in enumerate(rows):
            center_y = top - (row_index + 0.5) * band
            pdf.setFont(regular, 7.5)
            pdf.setFillColorRGB(*DARK)
            pdf.drawCentredString(x0 + panel_width / 2, center_y + 25, row["label"])
            for metric_index, (metric, color) in enumerate(zip(row["metrics"], (BLUE, ORANGE))):
                y = center_y + 7 - metric_index * 16
                mean, sd = metric["mean_change"], metric["sd_change"]
                end = x0 + mean / xmax * panel_width
                low = x0 + max(0.0, mean - sd) / xmax * panel_width
                high = x0 + min(xmax, mean + sd) / xmax * panel_width
                pdf.setFillColorRGB(*color)
                pdf.rect(x0, y - 4, max(0.5, end - x0), 8, fill=1, stroke=0)
                pdf.setStrokeColorRGB(*DARK)
                pdf.setLineWidth(0.6)
                pdf.line(low, y, high, y)
                pdf.line(low, y - 2, low, y + 2)
                pdf.line(high, y - 2, high, y + 2)
                pdf.setFont(bold, 6.8)
                pdf.setFillColorRGB(*DARK)
                label_x = min(max(end, high) + 3, x0 + panel_width - 22)
                pdf.drawString(label_x, y + 4.5, f"+{mean:.2f}")

        legend_labels = (
            ("Instance mIoU", "Category mIoU")
            if panel_index == 0
            else ("Clean accuracy", "Density accuracy")
        )
        legend_y = height - 36
        for index, (label, color) in enumerate(zip(legend_labels, (BLUE, ORANGE))):
            lx = x0 + 28 + index * 92
            pdf.setFillColorRGB(*color)
            pdf.rect(lx, legend_y - 4, 9, 6, fill=1, stroke=0)
            pdf.setFont(regular, 6.2)
            pdf.setFillColorRGB(*DARK)
            pdf.drawString(lx + 13, legend_y - 3, label)

        pdf.setFont(regular, 7)
        pdf.setFillColorRGB(*DARK)
        pdf.drawCentredString(x0 + panel_width / 2, 11, "Paired improvement (percentage points)")

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
