"""Automated evaluation: base model vs LoRA fine-tuned model on held-out
financial Q&A examples (real SEC 10-K excerpts the model never trained on).

For each example we generate an answer with both models and score:
  - rougeL                     text-overlap quality vs the gold answer
  - contains_gold_answer       did the model's output include the correct figure
  - numeric_hallucination_rate fraction of numbers stated that aren't in the source
  - latency_ms                 wall-clock generation time

This produces the "automated evaluation framework" and the hallucination-rate
number from the resume bullet. It compares fine-tuned vs its own base model
(TinyLlama), not GPT-3.5 — we have no OpenAI API key/budget in this environment.
The framework is written so a third model (e.g. GPT-3.5 via API) can be added
as another column in run_model() below without changing the metrics or report code.
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common import load_config, resolve_path, get_device
from src.evaluation.metrics import score_example


def load_eval_examples(cfg):
    path = resolve_path(cfg["dataset"]["prepared_dir"]) / "eval.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> tuple:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start) * 1000
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion.strip(), latency_ms


def run_model(model_id, adapter_path, examples, device, max_new_tokens, label):
    print(f"\n=== Generating with {label} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()

    results = []
    for i, ex in enumerate(examples):
        completion, latency_ms = generate(model, tokenizer, ex["prompt"], device, max_new_tokens)
        metrics = score_example(completion, ex["answer"], ex["context"])
        metrics.update({"latency_ms": latency_ms})
        results.append({
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "prediction": completion,
            **metrics,
        })
        print(f"[{label}] {i+1}/{len(examples)} rougeL={metrics['rougeL']:.2f} "
              f"hallucination={metrics['numeric_hallucination_rate']:.2f} "
              f"latency={latency_ms:.0f}ms")

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


def aggregate(results: list) -> dict:
    n = len(results)
    return {
        "n_examples": n,
        "avg_rougeL": sum(r["rougeL"] for r in results) / n,
        "avg_numeric_hallucination_rate": sum(r["numeric_hallucination_rate"] for r in results) / n,
        "gold_answer_recall": sum(r["contains_gold_answer"] for r in results) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
    }


def main():
    cfg = load_config()
    device = get_device()
    model_id = cfg["model"]["base_model_id"]
    adapter_path = str(resolve_path(cfg["training"]["output_dir"]) / "final")
    max_new_tokens = cfg["evaluation"]["max_new_tokens"]

    examples = load_eval_examples(cfg)
    print(f"Loaded {len(examples)} held-out eval examples | device={device}")

    base_results = run_model(model_id, None, examples, device, max_new_tokens, "base (pre-fine-tune)")
    ft_results = run_model(model_id, adapter_path, examples, device, max_new_tokens, "fine-tuned (LoRA)")

    report = {
        "base_model": aggregate(base_results),
        "fine_tuned_model": aggregate(ft_results),
    }
    base_h = report["base_model"]["avg_numeric_hallucination_rate"]
    ft_h = report["fine_tuned_model"]["avg_numeric_hallucination_rate"]
    if base_h > 0:
        report["hallucination_reduction_pct"] = round(100 * (base_h - ft_h) / base_h, 1)
    else:
        report["hallucination_reduction_pct"] = None

    report_path = resolve_path(cfg["evaluation"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    predictions_path = resolve_path(cfg["evaluation"]["predictions_path"])
    with open(predictions_path, "w") as f:
        for r in base_results:
            f.write(json.dumps({"model": "base", **r}) + "\n")
        for r in ft_results:
            f.write(json.dumps({"model": "fine_tuned", **r}) + "\n")

    print("\n=== Summary ===")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path} and {predictions_path}")


if __name__ == "__main__":
    main()
