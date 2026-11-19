"""Generates every architecture diagram in the README.   Run: python docs/diagrams/generate.py

Diagrams are code (Python `diagrams` library + Graphviz), so they stay in sync with the
system and reviews can diff them. Needs: pip install diagrams pillow; Graphviz `dot` on PATH.
Model nodes use small generated icons (the library ships none for scikit-learn / PyTorch).
"""
import tempfile
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECR, Fargate
from diagrams.aws.devtools import Codebuild
from diagrams.aws.management import Cloudformation, Cloudwatch
from diagrams.aws.ml import Bedrock
from diagrams.aws.network import ALB, CloudFront
from diagrams.aws.security import IAM, WAF, SecretsManager
from diagrams.custom import Custom
from diagrams.generic.storage import Storage
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import Users
from diagrams.onprem.container import Docker
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(exist_ok=True)
ICONS = Path(tempfile.mkdtemp(prefix="icons_"))
GRAPH = {"fontsize": "15", "bgcolor": "white", "pad": "0.4", "nodesep": "0.55", "ranksep": "0.9",
         "fontname": "Helvetica", "splines": "ortho"}
NODE = {"fontsize": "12", "fontname": "Helvetica"}


def icon(name: str, label: str, color: str) -> str:
    """Rounded square with a short text label, used as a node icon."""
    path = ICONS / f"{name}.png"
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16, 16, 240, 240), radius=44, fill=color)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 78 if len(label) <= 3 else 56)
    except OSError:
        font = ImageFont.load_default()
    w = d.textlength(label, font=font)
    d.text(((256 - w) / 2, 98), label, font=font, fill="white")
    img.save(path)
    return str(path)


BLUE, ORANGE, GREEN, VIOLET, GREY, RED = "#2a78d6", "#eb6834", "#1b8a5a", "#6250d6", "#52514e", "#c0392b"


def decision_pipeline() -> None:
    with Diagram("", filename=str(OUT / "decision_pipeline"), show=False, direction="LR", graph_attr=GRAPH | {"ranksep": "1.0", "nodesep": "0.7", "splines": "spline"},
                 node_attr=NODE, outformat="png"):
        feedback = Users("Customer feedback\n(review text)")
        bert = Custom("Fine-tuned DistilBERT\nint8 weights\nP(low), P(medium), P(high)", icon("bert", "BERT", ORANGE))
        with Cluster("Deterministic decision layer (decision.py)"):
            calib = Custom("Temperature\ncalibration", icon("cal", "T", GREEN))
            thresh = Custom("Thresholds\non probabilities", icon("thr", ">=", GREEN))
            rules = Custom("Review triggers\n(5 checks)", icon("rules", "!", RED))
        action = Custom("Action\nauto_reply\nmanual_review\nescalate", icon("act", "GO", GREY))
        with Cluster("Explanation only (cannot change the decision)"):
            llm = Bedrock("LLM explainer\nAmazon Bedrock\nNova Lite")
            val = Custom("Validators\nempty? contradicts?\ninvented numbers?", icon("val", "OK", VIOLET))
            fb = Custom("Local fallback\nexplanation", icon("fb", "FB", GREY))
        feedback >> bert >> calib >> thresh >> rules >> action
        action >> Edge(label="decision (read-only)") >> llm >> val
        val >> Edge(label="rejected twice", style="dashed") >> fb


def aws_architecture() -> None:
    small = {"width": "1.05", "height": "1.8"}   # smaller icons than the library default, for a tighter picture
    with Diagram("", filename=str(OUT / "aws_architecture"), show=False, direction="LR",
                 graph_attr=GRAPH | {"ranksep": "0.55", "nodesep": "0.3", "pad": "0.2", "fontsize": "13", "splines": "spline"},
                 node_attr=NODE, outformat="png"):
        client = Users("API client", **small)
        with Cluster("Edge"):
            waf = WAF("AWS WAF\nrate limit", **small)
            cdn = CloudFront("CloudFront\nHTTPS", **small)
        with Cluster("ECS on Fargate (1 to 3 tasks)"):
            alb = ALB("ALB\nCloudFront only", **small)
            api = Fargate("FastAPI container\nDistilBERT + decision", **small)
        bedrock = Bedrock("Bedrock\nNova Lite", **small)
        logs = Cloudwatch("CloudWatch\nmetrics, alarms", **small)
        secret = SecretsManager("Secrets Manager\nAPI key", **small)
        gha = GithubActions("GitHub Actions\nOIDC deploy", **small)
        client >> Edge(label="HTTPS + API key") >> waf >> cdn >> alb >> api
        api >> Edge(label="one model only") >> bedrock
        api >> Edge(label="metrics") >> logs
        secret >> Edge(style="dashed", label="at start") >> api
        gha >> Edge(style="dashed", label="CloudFormation") >> api


def release_path() -> None:
    with Diagram("", filename=str(OUT / "release_path"), show=False, direction="LR", graph_attr=GRAPH, node_attr=NODE, outformat="png"):
        gh = GithubActions("GitHub Actions\n(approved in the\nproduction environment)")
        with Cluster("Gates"):
            tests = Codebuild("93 tests")
            lint = Codebuild("cfn-lint")
        img = Docker("Build image\nlinux/amd64")
        ecr = ECR("ECR")
        cfn = Cloudformation("CloudFormation\nstack")
        svc = Fargate("ECS rolling\nupdate")
        smoke = Codebuild("Smoke test\nGET /")
        gh >> tests >> lint >> img >> ecr >> cfn >> svc >> smoke
        IAM("OIDC role\nno stored keys") >> Edge(style="dashed") >> gh


def training_pipeline() -> None:
    with Diagram("", filename=str(OUT / "training_pipeline"), show=False, direction="LR", graph_attr=GRAPH | {"ranksep": "0.9", "nodesep": "0.7", "splines": "spline"},
                 node_attr=NODE, outformat="png"):
        raw = Storage("Women's E-Commerce\nClothing Reviews\n(rating -> tier)")
        splits = Python("make_splits.py\n70 / 15 / 15\nstratified")
        syn = Bedrock("generate_synthetic.py\nLLM-written sarcasm\nand negation")
        with Cluster("Train (train split only)"):
            bert = Custom("train_transformer_model.py\nDistilBERT, 2 epochs", icon("t2", "BERT", ORANGE))
            q = Custom("compact_transformer.py\nint8, 60 MB", icon("t3", "int8", ORANGE))
            base = Custom("train_classical_model.py\n(baseline only)", icon("t1", "LR", GREY))
        tune = Custom("tune_decision.py\nvalidation split only:\ntemperature, thresholds", icon("t4", "VAL", GREEN))
        with Cluster("Evaluate (test split, touched once)"):
            ev = Custom("evaluate_hybrid.py\nevaluate_robustness.py\nevaluate_behaviour.py\nevaluate_llm.py", icon("t5", "TEST", VIOLET))
        art = Storage("models/\ntransformer_assets/\ndecision_config.json")
        raw >> splits >> bert >> q >> tune >> ev
        syn >> Edge(style="dashed", label="extra train data") >> bert
        splits >> Edge(style="dashed") >> base
        tune >> art


def main() -> None:
    decision_pipeline()
    aws_architecture()
    release_path()
    training_pipeline()
    print("wrote diagrams to", OUT)


if __name__ == "__main__":
    main()
