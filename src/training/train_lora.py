"""LoRA supervised fine-tuning.

Model and hyperparameters come from config.yaml. Swapping the base model to
mistralai/Mistral-7B-v0.1 on a CUDA GPU requires no changes to this file —
only config.yaml (base_model_id, quantization.load_in_4bit, and probably
larger max_train_examples / max_steps).
"""
import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common import load_config, resolve_path, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override config's training.max_steps (useful for a quick smoke test)")
    args = parser.parse_args()

    cfg = load_config()
    model_id = cfg["model"]["base_model_id"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]

    device = get_device()
    print(f"Device: {device} | Base model: {model_id}")

    prepared_dir = resolve_path(ds_cfg["prepared_dir"])
    train_ds = load_dataset("json", data_files=str(prepared_dir / "train.jsonl"), split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # fp32 is the safe/portable choice on MPS and CPU
    )
    model.to(device)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = resolve_path(train_cfg["output_dir"])
    max_steps = args.max_steps if args.max_steps is not None else train_cfg["max_steps"]

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        weight_decay=train_cfg["weight_decay"],
        seed=train_cfg["seed"],
        max_length=ds_cfg["max_seq_length"],
        report_to=[],
        use_mps_device=(device == "mps"),
        save_strategy="steps",
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
