"""Shared helpers: config loading, device selection, prompt formatting.

Every other script imports from here so the "how do we format a financial
Q&A example into a prompt" logic exists in exactly one place.
"""
import yaml
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = None) -> dict:
    path = path or (REPO_ROOT / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def get_device() -> str:
    """CUDA > MPS (Apple GPU) > CPU. TinyLlama/Mistral both run under this same call."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using only "
    "the information in the provided filing excerpt. Be concise and cite "
    "numbers exactly as given."
)


def build_prompt(question: str, context: str) -> str:
    """Instruction-style prompt shared by training, eval, and serving so the
    fine-tuned model always sees the same shape of input it was trained on."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\nFiling excerpt:\n{context}\n\nQuestion: {question}</s>\n"
        f"<|assistant|>\n"
    )
