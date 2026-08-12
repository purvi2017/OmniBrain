"""
render_screenshot.py
---------------------
Utility to render captured terminal text output into a PNG image
that looks like a terminal screenshot, so test results can be shared
as images (e.g. for the team chat / PR review).

Usage:
    python tools/render_screenshot.py <input_txt> <output_png> <title>
"""

import sys
from PIL import Image, ImageDraw, ImageFont


def render_terminal_screenshot(text_path: str, out_path: str, title: str = "Terminal"):
    with open(text_path, "r") as f:
        lines = f.read().splitlines()

    font_size = 15
    line_height = 20
    padding = 20
    header_height = 40
    char_width = 9

    max_line_len = max((len(line) for line in lines), default=40)
    width = max(700, padding * 2 + max_line_len * char_width)
    height = header_height + padding * 2 + line_height * len(lines)

    bg_color = (30, 30, 30)
    header_color = (50, 50, 50)
    text_color = (0, 230, 120)
    title_color = (220, 220, 220)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # header bar
    draw.rectangle([0, 0, width, header_height], fill=header_color)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([15 + i * 22, 13, 29 + i * 22, 27], fill=c)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.text((90, 11), title, fill=title_color, font=title_font)

    y = header_height + padding
    for line in lines:
        color = text_color
        if line.startswith("[PASS]"):
            color = (80, 250, 123)
        elif line.startswith("[WARN]"):
            color = (255, 184, 108)
        elif line.startswith("[FAIL]") or "FAILED" in line:
            color = (255, 85, 85)
        elif line.startswith("RESULT"):
            color = (139, 233, 253)
        elif line.startswith("="):
            color = (98, 114, 164)
        draw.text((padding, y), line, fill=color, font=font)
        y += line_height

    img.save(out_path)
    print(f"Saved screenshot: {out_path}")


if __name__ == "__main__":
    text_path, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3]
    render_terminal_screenshot(text_path, out_path, title)
