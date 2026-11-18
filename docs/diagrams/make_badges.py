"""Writes the README's shields-style SVG badges to docs/images/badges/ (no external service needed).

    python docs/diagrams/make_badges.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images" / "badges"
OUT.mkdir(parents=True, exist_ok=True)

# name, label, value, value colour (all dark enough for white text)
BADGES = [
    ("python", "Python", "3.11", "#3776AB"),
    ("pytorch", "PyTorch", "DistilBERT", "#C2361A"),
    ("scikit-learn", "scikit-learn", "TF-IDF + LogReg", "#B8650A"),
    ("fastapi", "FastAPI", "serving", "#00796B"),
    ("aws", "AWS", "ECS Fargate", "#B26A00"),
    ("bedrock", "Amazon Bedrock", "explanations", "#0F7B6C"),
    ("cloudfront", "CloudFront + WAF", "HTTPS, rate limit", "#6B3FA0"),
    ("cloudformation", "CloudFormation", "one stack", "#C2185B"),
    ("docker", "Docker", "550 MB in ECR", "#1D6FB8"),
    ("github-actions", "GitHub Actions", "CI + OIDC deploy", "#2F3A4A"),
    ("tests", "tests", "68 passing", "#2E7D32"),
    ("deterministic", "decisions", "deterministic", "#4F46E5"),
    ("llm-explains-only", "LLM", "explains only", "#7B3FC4"),
]


def width(text: str) -> int:
    return int(len(text) * 6.4) + 12


def svg(label: str, value: str, color: str) -> str:
    lw, vw = width(label), width(value)
    tw = lw + vw
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{tw}" height="20" role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <clipPath id="r"><rect width="{tw}" height="20" rx="3"/></clipPath>
  <g clip-path="url(#r)"><rect width="{lw}" height="20" fill="#555"/><rect x="{lw}" width="{vw}" height="20" fill="{color}"/></g>
  <g font-family="Verdana,DejaVu Sans,Geneva,sans-serif" font-size="11" fill="#fff" text-anchor="middle">
    <text x="{lw / 2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{lw / 2}" y="14">{label}</text>
    <text x="{lw + vw / 2}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{lw + vw / 2}" y="14">{value}</text>
  </g>
</svg>
'''


for name, label, value, color in BADGES:
    (OUT / f"{name}.svg").write_text(svg(label, value, color))
print("wrote", len(BADGES), "badges to", OUT)
