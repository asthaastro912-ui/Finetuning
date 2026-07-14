import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import (
    numeric_hallucination_rate,
    contains_gold_answer,
    rouge_l,
)


def test_numeric_hallucination_rate_flags_fabricated_number():
    context = "Revenue was $2,299,887 thousand in fiscal 2023."
    grounded = "Revenue was $2,299,887 thousand."
    fabricated = "Revenue was $9,000,000 thousand."
    assert numeric_hallucination_rate(grounded, context) == 0.0
    assert numeric_hallucination_rate(fabricated, context) == 1.0


def test_numeric_hallucination_rate_no_numbers_is_zero():
    assert numeric_hallucination_rate("Revenue was not disclosed.", "some context") == 0.0


def test_contains_gold_answer_numeric():
    assert contains_gold_answer("The debt was $2,299,887 thousand", "$2,299,887 thousand")
    assert not contains_gold_answer("The debt was $1 thousand", "$2,299,887 thousand")


def test_rouge_l_identical_is_one():
    assert rouge_l("the sky is blue", "the sky is blue") == 1.0
