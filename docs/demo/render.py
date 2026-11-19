"""
Turns recordings into the README images, with agg (asciinema's renderer). Needs Pillow.

    python docs/demo/render.py demo               # casts/tiers.cast + casts/fallback.cast -> docs/images/demo.gif
    python docs/demo/render.py still ecs infra_ecs.png    # last frame of casts/ecs.cast -> docs/images/infra_ecs.png

merge_casts joins recordings made at different times into one timeline (the fallback clip needs the
service reconfigured in between).
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[2]
CASTS, IMAGES = ROOT / "docs" / "demo" / "casts", ROOT / "docs" / "images"
AGG = ["agg", "--theme", "github-dark", "--font-size", "16", "--idle-time-limit", "2", "--last-frame-duration", "4"]


def merge_casts(parts, out, gap=1.0):
    header, events, offset = None, [], 0.0
    for p in parts:
        lines = [json.loads(l) for l in open(p) if l.strip()]
        header = header or lines[0]
        evs = lines[1:]
        events += [[round(e[0] + offset, 6), *e[1:]] for e in evs]
        offset = (events[-1][0] if events else 0.0) + gap
    with open(out, "w") as f:
        f.write(json.dumps(header) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")


def gif(cast, out):
    subprocess.run([*AGG, str(cast), str(out)], check=True)
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    if sys.argv[1] == "demo":
        merged = CASTS / "demo.cast"
        merge_casts([CASTS / "tiers.cast", CASTS / "fallback.cast"], merged)
        gif(merged, IMAGES / "demo.gif")
    elif sys.argv[1] == "still":
        name, out = sys.argv[2], sys.argv[3]
        tmp = CASTS / f"{name}.gif"
        gif(CASTS / f"{name}.cast", tmp)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(tmp))]
        frames[-1].save(IMAGES / out)
        tmp.unlink()
        print(f"wrote {IMAGES / out}")
