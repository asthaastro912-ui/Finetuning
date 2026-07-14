"""Automated metrics for comparing base vs fine-tuned model outputs.

Three signal types, deliberately kept dependency-light (no external judge
model, so this runs fully offline/free):
  - rougeL: standard text-overlap quality metric against the gold answer.
  - numeric_hallucination: for financial QA the answer is almost always a
    number/date pulled from the context. We extract every number in the
    model's output and check it appears in the source context. A number
    that appears in the answer but NOT in the context is a fabricated
    figure — a direct, cheap proxy for hallucination in this domain.
  - exact/contains_match: whether the gold answer's key figure shows up in
    the prediction at all (a recall-style correctness signal).
"""
import re
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def rouge_l(prediction: str, reference: str) -> float:
    return _scorer.score(reference, prediction)["rougeL"].fmeasure


def extract_numbers(text: str) -> set:
    found = set()
    for match in NUMBER_RE.findall(text):
        cleaned = match.strip("$%").replace(",", "")
        try:
            found.add(round(float(cleaned), 2))
        except ValueError:
            continue
    return found


def numeric_hallucination_rate(prediction: str, context: str) -> float:
    """Fraction of numbers in the prediction that do NOT appear in the source context.
    0.0 = every number the model stated is grounded in the filing excerpt."""
    pred_nums = extract_numbers(prediction)
    if not pred_nums:
        return 0.0
    context_nums = extract_numbers(context)
    fabricated = [n for n in pred_nums if n not in context_nums]
    return len(fabricated) / len(pred_nums)


def contains_gold_answer(prediction: str, reference: str) -> bool:
    ref_nums = extract_numbers(reference)
    if ref_nums:
        pred_nums = extract_numbers(prediction)
        return bool(ref_nums & pred_nums)
    # non-numeric answer: fall back to substring containment
    return reference.strip().lower() in prediction.strip().lower()


def score_example(prediction: str, reference: str, context: str) -> dict:
    return {
        "rougeL": rouge_l(prediction, reference),
        "numeric_hallucination_rate": numeric_hallucination_rate(prediction, context),
        "contains_gold_answer": contains_gold_answer(prediction, reference),
    }
