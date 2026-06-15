import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from engine.core import config
from engine.finetuning.dataset import build_chat_example

logger = logging.getLogger("engine.train_lora")


def _select_runtime(preferred_device: str) -> tuple[str, torch.dtype, bool]:
    preferred = str(preferred_device or "auto").strip().lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported --device value: {preferred_device}")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested, but CUDA is not available.")
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16, True
        return "cuda", torch.float16, True
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested, but MPS is not available.")
        return "mps", torch.float16, False
    if preferred == "cpu":
        return "cpu", torch.float32, False

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16, True
        return "cuda", torch.float16, True
    if torch.backends.mps.is_available():
        return "mps", torch.float16, False
    return "cpu", torch.float32, False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for LegalRAG on JSONL instruction data."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset with instruction/context/response fields.",
    )
    parser.add_argument(
        "--base-model",
        default=config.HF_LLM_BASE_MODEL,
        help="Base HuggingFace causal LM.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the LoRA adapter and tokenizer.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--force-device-map-auto",
        action="store_true",
        help="Force device_map='auto'. Use only on multi-device setups where accelerate mapping is desired.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def _build_dataset(records: list[dict], tokenizer, max_length: int) -> Dataset:
    prompt_ids_list: list[list[int]] = []
    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    labels_list: list[list[int]] = []

    for record in records:
        prompt_text, answer_text = build_chat_example(record)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        eos_id = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        input_ids = (prompt_ids + response_ids + eos_id)[:max_length]
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids) + response_ids + eos_id)[:max_length]

        prompt_ids_list.append(prompt_ids[:max_length])
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )


def _move_model_to_runtime(model, runtime_device: str):
    if runtime_device == "mps":
        try:
            return model.to("mps")
        except RuntimeError as exc:
            message = str(exc)
            if "MPS backend out of memory" in message:
                raise RuntimeError(
                    "The base model does not fit in Apple MPS memory. "
                    "Use a smaller base model on this Mac (for example Qwen/Qwen2.5-1.5B-Instruct), "
                    "or rerun with --device cpu, or train on a CUDA machine with --load-in-4bit."
                ) from exc
            raise
    if runtime_device == "cpu":
        return model.to("cpu")
    return model


def _enable_input_grads(model) -> None:
    # Gradient checkpointing on frozen-base LoRA needs the input activations
    # to stay attached to the autograd graph.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return

    def _require_grads(_module, _inputs, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(_require_grads)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    config.configure_hf_hub()
    runtime_device, runtime_dtype, can_use_kbit = _select_runtime(args.device)

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "trust_remote_code": True,
        "cache_dir": config.HF_CACHE_DIR,
    }
    if config.HF_TOKEN:
        common_kwargs["token"] = config.HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **common_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(common_kwargs)
    model_kwargs["dtype"] = runtime_dtype
    if args.force_device_map_auto or runtime_device == "cuda":
        model_kwargs["device_map"] = "auto"
    if args.load_in_4bit:
        if not can_use_kbit:
            raise ValueError(
                "--load-in-4bit requires CUDA. On macOS/CPU run without --load-in-4bit."
            )
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    model = _move_model_to_runtime(model, runtime_device)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _enable_input_grads(model)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_records = _load_jsonl(dataset_path)
    train_dataset = _build_dataset(raw_records, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        bf16=runtime_device == "cuda" and runtime_dtype == torch.bfloat16,
        fp16=runtime_device == "cuda" and runtime_dtype == torch.float16,
        use_mps_device=runtime_device == "mps",
        no_cuda=runtime_device == "cpu",
        dataloader_pin_memory=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Saved LoRA adapter to %s", output_dir)
    logger.info("Training runtime: device=%s dtype=%s", runtime_device, runtime_dtype)
    logger.info(
        "Use with LEGAL_RAG_LLM_BACKEND=hf_peft, LEGAL_RAG_HF_BASE_MODEL=%s, LEGAL_RAG_HF_ADAPTER_PATH=%s",
        args.base_model,
        output_dir,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
