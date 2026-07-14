"""FastAPI inference service for the fine-tuned financial-QA model.

Loads the base model once at startup, merges in the LoRA adapter (falls back
to the base model with a warning if no adapter has been trained yet), and
serves /generate. Every request/response is logged to SQLite so the
monitoring dashboard has real data to show.

Run: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common import load_config, resolve_path, get_device, build_prompt
from src.evaluation.metrics import numeric_hallucination_rate
from src.monitoring.db import get_connection, log_request
from src.serving.schemas import GenerateRequest, GenerateResponse, HealthResponse

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    device = get_device()
    model_id = cfg["model"]["base_model_id"]
    adapter_dir = resolve_path(cfg["training"]["output_dir"]) / "final"

    print(f"Loading base model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    adapter_loaded = adapter_dir.exists()
    if adapter_loaded:
        print(f"Merging LoRA adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
    else:
        print(f"WARNING: no adapter found at {adapter_dir}, serving the base model unmodified.")

    model.to(device)
    model.eval()

    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    STATE["device"] = device
    STATE["model_id"] = model_id
    STATE["adapter_loaded"] = adapter_loaded
    STATE["max_new_tokens"] = cfg["serving"]["max_new_tokens"]
    STATE["db"] = get_connection(str(resolve_path(cfg["serving"]["log_db_path"])))

    yield
    STATE.clear()


app = FastAPI(title="Financial QA LoRA Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=STATE["device"],
        model_id=STATE["model_id"],
        adapter_loaded=STATE["adapter_loaded"],
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    device = STATE["device"]
    max_new_tokens = req.max_new_tokens or STATE["max_new_tokens"]

    prompt = build_prompt(req.question, req.context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        latency_ms = (time.time() - start) * 1000
        h_rate = numeric_hallucination_rate(answer, req.context)

        log_request(STATE["db"], req.question, len(req.context), answer, latency_ms, h_rate)
        return GenerateResponse(answer=answer, latency_ms=latency_ms, numeric_hallucination_rate=h_rate)
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        log_request(STATE["db"], req.question, len(req.context), "", latency_ms,
                     status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
