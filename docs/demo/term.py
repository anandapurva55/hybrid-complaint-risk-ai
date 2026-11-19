"""Tiny terminal renderer (Pillow) shared by the demo GIF and the screenshots. Account IDs are redacted."""
import re

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Menlo.ttc"
BG, FG, DIM = (13, 17, 23), (201, 209, 217), (125, 133, 144)
GREEN, RED, AMBER, BLUE = (63, 185, 80), (248, 81, 73), (210, 153, 34), (121, 192, 255)
ACCOUNT_ID = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")


def redact(text):
    """Replace 12-digit AWS account IDs with block characters."""
    return ACCOUNT_ID.sub("█" * 12, text)


class Terminal:
    def __init__(self, title, cols=120, rows=14, size=16, lh=25, pad=28):
        self.font = ImageFont.truetype(FONT, size)
        self.cw, self.lh, self.pad, self.title = self.font.getlength("M"), lh, pad, title
        self.W = int(cols * self.cw + 2 * pad)
        self.H = 38 + 24 + rows * lh + pad
        self.cols = cols

    def frame(self, lines, cursor_at=None):
        """lines: list of segment lists [(text, colour), ...]. cursor_at: (line_index, column) or None."""
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, self.W, 38), fill=(22, 27, 34))
        for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse((18 + i * 22, 12, 30 + i * 22, 24), fill=c)
        d.text((self.W / 2 - self.font.getlength(self.title) / 2, 10), self.title, font=self.font, fill=DIM)
        y = 62
        for n, segs in enumerate(lines):
            x = self.pad
            for text, col in segs:
                text = redact(text)
                d.text((x, y), text, font=self.font, fill=col)
                x += self.font.getlength(text)
            if cursor_at and cursor_at[0] == n:
                cx = self.pad + cursor_at[1] * self.cw
                d.rectangle((cx, y + 2, cx + self.cw - 1, y + 19), fill=FG)
            y += self.lh
        return img


def wrap_command(command, cols):
    """Wrap a long shell command like a terminal would, continuing with two spaces."""
    out, first = [], True
    while command:
        room = cols - 2 - (0 if first else 2)
        out.append(command[:room]); command = command[room:]; first = False
    return out
