"""Download the public SEC 10-K financial Q&A dataset and turn it into
train/eval splits in the instruction format the training and eval scripts use.

Dataset: virattt/financial-qa-10K on the HF Hub — ~7,000 (question, answer,
context, ticker, filing) rows, where `context` is an excerpt straight out of
a company's real SEC 10-K regulatory filing and `answer` is grounded in it.
That grounding is what lets the eval framework later check for hallucination:
we know the exact source text each answer should have come from.
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common import load_config, resolve_path, build_prompt


def main():
    cfg = load_config()
    ds_cfg = cfg["dataset"] 
    ## ds_cfg["hf_dataset_id"] is "virattt/financial-qa-10K"
    print(f"Loading {ds_cfg['hf_dataset_id']} from the Hugging Face Hub...") 
    raw = load_dataset(ds_cfg["hf_dataset_id"], split="train")
    ## raw is a dataset object (lazy memory efficient table)
    print(f"Raw rows: {len(raw)} | columns: {raw.column_names}")


    raw = raw.shuffle(seed=ds_cfg["seed"])
    ## Why shuffle first? The raw dataset might be ordered in some way we don't 
    # want (e.g., grouped by company or filing year) — if we just sliced the first 
    # 90% without shuffling, our eval set could end up being "all one company" or 
    # "all one year," which wouldn't be a fair, representative test. Shuffling with 
    # a fixed seed=42 randomizes the order reproducibly — anyone who runs this script 
    # gets the exact same shuffle, so results are comparable across runs/people.
    n_total = len(raw)
    n_train_pool = int(n_total * ds_cfg["train_fraction"])
    train_pool = raw.select(range(n_train_pool))
    eval_pool = raw.select(range(n_train_pool, n_total))
    #the mental model of "eval is a fixed, untouched holdout region of the full dataset" 
    # stays correct even if you later change max_train_examples to use more data.
    max_train = ds_cfg.get("max_train_examples")
    if max_train:
        train_pool = train_pool.select(range(min(max_train, len(train_pool))))
    max_eval = ds_cfg.get("max_eval_examples")
    if max_eval:
        eval_pool = eval_pool.select(range(min(max_eval, len(eval_pool))))

    print(f"Using {len(train_pool)} train examples, {len(eval_pool)} eval examples")

    out_dir = resolve_path(ds_cfg["prepared_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_records(split):
        records = []
        for row in split:
            prompt = build_prompt(row["question"], row["context"])
            records.append({
                "question": row["question"],
                "context": row["context"],
                "answer": row["answer"].strip(),
                "ticker": row.get("ticker"),
                "filing": row.get("filing"),
                "prompt": prompt,
                # trl's SFTTrainer switches to prompt-completion mode (loss masked
                # on the prompt, trained only on the completion) whenever a
                # "prompt" key is present and a matching "completion" key exists.
                "completion": " " + row["answer"].strip(),
                #here's a leading space before the answer. This matters because of 
                # how tokenizers work: "$2,299,887" and " $2,299,887" can tokenize 
                # into different token sequences (many tokenizers have "space-prefixed"
                #  versions of common tokens). Since the prompt ends in <|assistant|>\n 
                # with no trailing space, adding that space at the start of the completion 
                # produces the natural token boundary the model would generate anyway — 
                # without it, training could inadvertently teach the model a slightly 
                # unnatural token sequence.
            })
        return records

    train_records = to_records(train_pool)
    eval_records = to_records(eval_pool)

    with open(out_dir / "train.jsonl", "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "eval.jsonl", "w") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    ctx_lens = [len(r["context"].split()) for r in train_records]
    print(f"Context length (words): min={min(ctx_lens)} "
          f"median={sorted(ctx_lens)[len(ctx_lens)//2]} max={max(ctx_lens)}")
    print(f"Wrote {out_dir/'train.jsonl'} and {out_dir/'eval.jsonl'}")


if __name__ == "__main__":
    main()
