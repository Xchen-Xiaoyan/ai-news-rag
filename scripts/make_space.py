"""Assemble the Hugging Face Space folder (code only; data comes from the private dataset at runtime).

  python scripts/make_space.py
  hf upload <user>/ai-news-rag build/space . --repo-type space   (Docker Space)
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", OUT / "src", ignore=ignore)
    shutil.copytree(ROOT / "app", OUT / "app", ignore=ignore)
    shutil.copytree(ROOT / ".streamlit", OUT / ".streamlit")
    shutil.copy(ROOT / "deploy" / "requirements.txt", OUT / "requirements.txt")
    shutil.copy(ROOT / "deploy" / "SPACE_README.md", OUT / "README.md")
    shutil.copy(ROOT / "deploy" / "Dockerfile", OUT / "Dockerfile")
    print("\n".join(str(p.relative_to(OUT)) for p in sorted(OUT.rglob("*")) if p.is_file()))


if __name__ == "__main__":
    main()
