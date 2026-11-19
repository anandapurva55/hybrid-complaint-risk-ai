"""
Records the terminal demo shown in the README (docs/images/demo.gif).

    HYBRID_RISK_API_KEY=... python docs/demo/record.py --capture https://<distribution>.cloudfront.net [--clips 0,1,2] [--merge]
    python docs/demo/record.py --render        # re-render from docs/demo/captured.json, runs nothing

--capture runs the REAL command `python -m hybrid_risk --url <service> "<text>"` for each clip and stores
its output with the capture date. Only the typing animation is synthetic. The service in the shipped GIF
was a live deployment captured on the date shown in the title bar and has since been torn down.
"""
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image  # noqa: E402

from term import AMBER, BLUE, DIM, FG, GREEN, RED, Terminal  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "docs" / "demo" / "captured.json"
OUT = ROOT / "docs" / "images" / "demo.gif"

CLIPS = [  # (text, comment shown above the command)
    ("Love this sweater, perfect fit and gorgeous color.", "# 1. clearly positive: answered automatically"),
    ("Pretty dress but the material is not as soft as I hoped.", "# 2. mixed: a person should look"),
    ("Fell apart in the wash. Worst purchase ever, I want a refund!", "# 3. angry: escalate"),
    ("Fell apart in the wash. Worst purchase ever, I want a refund!", "# 4. same request with the LLM unavailable: same decision, local fallback explanation"),
]
TIER = {"high": RED, "medium": AMBER, "low": GREEN}
ACTION = {"escalate_immediately": RED, "manual_review": AMBER, "auto_reply": GREEN}


def capture(url, indices, merge=False):
    """Run the real command for the given clip indices and store the output (merged into an earlier capture)."""
    data = json.loads(CAPTURE.read_text()) if merge and CAPTURE.exists() else {"captured": str(date.today()), "rows": [None] * len(CLIPS)}
    for idx in indices:
        text, comment = CLIPS[idx]
        cmd = [sys.executable, "-m", "hybrid_risk", "--url", url, text]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=os.environ).stdout.rstrip("\n")
        data["rows"][idx] = {"comment": comment, "command": f'python -m hybrid_risk --url {url} "{text}"', "output": out.split("\n")}
        print(out, end="\n\n")
    CAPTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def display(command):
    """The command as a person would type it: a backslash continuation before the quoted feedback."""
    head, quoted = command.split(' "', 1)
    return [head + " \\", '    "' + quoted]


def colour(line):
    label, value = line[:13], line[13:]
    first = value.split()[0] if value.split() else ""
    if label.startswith("Tier"):
        return [(label, DIM), (first, TIER.get(first, FG)), (value[len(first):], DIM)]
    if label.startswith("Action"):
        return [(label, DIM), (first, ACTION.get(first, FG)), (value[len(first):], DIM)]
    if label.startswith("Feedback"):
        return [(label, DIM), (value, BLUE)]
    if label.startswith("Source"):
        return [(label, DIM), (value, AMBER if value.strip() == "fallback" else FG)]
    return [(label, DIM), (value, FG)]


def render():
    data = json.loads(CAPTURE.read_text())
    term = Terminal(f"hybrid-risk: live deployment, captured {data['captured']} (since torn down)", cols=124, rows=14)
    frames, durations = [], []
    add = lambda img, ms: (frames.append(img), durations.append(ms))
    add(term.frame([]), 600)
    for row in data["rows"]:
        full = "\n".join(display(row["command"]))
        head = [[(row["comment"], DIM)]]
        mk = lambda txt: head + [[("$ " if i == 0 else "  ", GREEN), (t, FG)] for i, t in enumerate(txt.split("\n"))]
        for n in range(0, len(full) + 1, 7):
            lines = mk(full[:n])
            add(term.frame(lines, cursor_at=(len(lines) - 1, 2 + len(lines[-1][1][1]))), 40)
        prompt = mk(full)
        add(term.frame(prompt), 450)
        for k in range(1, len(row["output"]) + 1):
            add(term.frame(prompt + [colour(l) for l in row["output"][:k]]), 150 if k > 1 else 400)
        add(term.frame(prompt + [colour(l) for l in row["output"]]), 3000)
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=2)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    if "--capture" in sys.argv:
        idx = [int(i) for i in sys.argv[sys.argv.index("--clips") + 1].split(",")] if "--clips" in sys.argv else list(range(len(CLIPS)))
        capture(sys.argv[sys.argv.index("--capture") + 1], idx, merge="--merge" in sys.argv)
    if "--no-render" not in sys.argv and all(r for r in json.loads(CAPTURE.read_text())["rows"]):
        render()
