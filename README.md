# Financial QA LoRA Fine-Tuning Pipeline

An end-to-end, runnable version of the "fine-tuned an LLM on financial
regulatory documents" project: data prep → LoRA supervised fine-tuning →
automated evaluation → FastAPI serving → live monitoring dashboard.

## Honest scope of the demo

Two things are substituted here relative to a literal reading of the resume
bullet, both driven by hardware:

| Resume claim | This repo | Why |
|---|---|---|
| Mistral-7B | **TinyLlama-1.1B-Chat** | Fine-tuning happens on a 16GB Apple Silicon Mac with no CUDA GPU. 7B params alone is ~14GB in fp16 before gradients/optimizer state — not safe on 16GB unified memory. TinyLlama is the same causal-LM/LoRA code path; `config.yaml`'s `model.base_model_id` is the only line that needs to change to target Mistral-7B on a real GPU (plus flipping `quantization.load_in_4bit: true`, which needs `bitsandbytes` + CUDA). |
| "vs GPT-3.5" hallucination comparison | **base TinyLlama vs fine-tuned TinyLlama** | No OpenAI API key/budget in this environment. `src/evaluation/evaluate.py` is written so a third model (GPT-3.5 via API) can be added as another `run_model()`-style column without touching the metrics code. |

Everything else — the SEC-filing dataset, LoRA training, the eval framework,
the FastAPI service, request logging, and the monitoring dashboard — is real
and actually runs, not mocked.

Apple Silicon note: there's no CUDA GPU here, but PyTorch's `mps` backend
uses the M-series GPU directly, which is meaningfully faster than plain CPU.
`src/common.py:get_device()` picks CUDA > MPS > CPU automatically.

## Dataset

[`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K)
on the Hugging Face Hub — ~7,000 (question, answer, context, ticker, filing)
rows, where `context` is an excerpt taken directly from a company's real SEC
10-K regulatory filing and `answer` is grounded in that excerpt. That
grounding is also what the hallucination metric relies on (see below).

## Pipeline

```
src/data/prepare_dataset.py   downloads the dataset, builds train/eval splits
                               in an instruction (prompt, completion) format
src/training/train_lora.py    LoRA SFT via trl's SFTTrainer + peft
src/evaluation/metrics.py     rougeL, numeric hallucination rate, answer recall
src/evaluation/evaluate.py    runs base vs fine-tuned model over held-out
                               examples, writes artifacts/eval_report.json
src/serving/app.py            FastAPI service: loads base model + merges the
                               LoRA adapter, exposes POST /generate
src/monitoring/db.py          SQLite request log shared by the API and dashboard
src/monitoring/dashboard.py   Streamlit dashboard: live request volume,
                               latency, hallucination rate + the offline eval report
```

All hyperparameters (model id, LoRA rank, batch size, step count, eval sample
size, ports, etc.) live in `config.yaml` — nothing is hardcoded in the scripts.

## Why these specific eval metrics

Financial QA answers are almost always a number or date pulled from the
filing excerpt. That makes hallucination cheap to detect *without* an LLM
judge: extract every number the model states, check whether it appears
in the source context.

- **`numeric_hallucination_rate`** — fraction of numbers in the model's
  answer that do NOT appear in the context. 0.0 = every figure is grounded.
- **`contains_gold_answer`** — did the model's output include the correct
  figure (or, for non-numeric answers, the gold text)?
- **`rougeL`** — standard text-overlap quality metric vs. the gold answer.
- **`latency_ms`** — wall-clock generation time per request.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Prepare data (downloads virattt/financial-qa-10K, ~3MB)
python -m src.data.prepare_dataset

# 2. Fine-tune (LoRA adapter saved to artifacts/lora-adapter/final)
python -m src.training.train_lora
#   quick smoke test instead of the full 150-step run:
#   python -m src.training.train_lora --max_steps 5

# 3. Evaluate base vs fine-tuned on held-out examples
python -m src.evaluation.evaluate

# 4. Serve (binds to localhost only; pass --host 0.0.0.0 to expose on the network)
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d \
  '{"question": "What was total revenue?", "context": "Total revenue was $1.2 billion in fiscal 2023."}'

# 5. Monitor (separate terminal, while the API is running)
streamlit run src/monitoring/dashboard.py --server.address 127.0.0.1
```

## Results (this actual run)

40 held-out SEC 10-K questions the model never trained on, TinyLlama-1.1B base vs. LoRA fine-tuned, 150 training steps / 1200 examples, on an M4 Mac via PyTorch MPS:

| Metric | Base model | Fine-tuned (LoRA) |
|---|---|---|
| ROUGE-L (vs gold answer) | 0.451 | **0.661** |
| Numeric hallucination rate | 0.090 | **0.025** |
| Gold answer recall | 0.600 | **0.625** |
| Avg latency | 5616 ms* | 3528 ms* |

**Hallucination rate reduction: 72.2%** — the fine-tuned model fabricates far fewer numbers not present in the source filing excerpt.

\* Both eval runs executed back-to-back on shared hardware; treat latency as directional, not a clean benchmark. Standalone (no contention), the fine-tuned model answers in ~0.5–0.8s per request (see the live dashboard).

Full per-example predictions: `artifacts/eval_predictions.jsonl`. Full aggregate report: `artifacts/eval_report.json`.

## Scaling this up for real

- **Full dataset / longer training**: set `dataset.max_train_examples: null`
  and raise/remove `training.max_steps` in `config.yaml`.
- **Mistral-7B on a CUDA GPU**: change `model.base_model_id` to
  `mistralai/Mistral-7B-v0.1`, set `quantization.load_in_4bit: true`, install
  `bitsandbytes`, and load the model with `BitsAndBytesConfig` in
  `train_lora.py` / `evaluate.py` (4-bit QLoRA is the standard way to fit a
  7B model's fine-tuning in consumer GPU memory).
- **Real hallucination baseline vs GPT-3.5**: add an OpenAI call as a third
  `run_model()`-style function in `evaluate.py` and include it in the report.
- **Multi-worker serving**: swap the SQLite request log for Postgres once
  the FastAPI service runs as more than one process.
