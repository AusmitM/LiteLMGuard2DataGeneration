"""
Few-shot data transformation with Qwen3.5-9B-Base (8-bit, bitsandbytes).

You give it:
  - examples.jsonl : worked examples, one per line: {"input": "...", "output": "..."}
  - data.jsonl     : items to transform, one per line: {"input": "..."}
                     (a plain .txt file with one item per line also works)
It writes:
  - results.jsonl  : {"input": "...", "output": "..."} for every item

Usage:
  python fewshot_transform.py --examples examples.jsonl --data data.jsonl --out results.jsonl
Optional:
  --instruction "Rewrite each sentence in formal English."   (a one-line task description at the top of the prompt)
  --batch-size 4  --max-new-tokens 256  --model Qwen/Qwen3.5-9B-Base  --4bit
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# bitsandbytes 8-bit layers always compute in float16 and complain on every layer that
# receives bfloat16 input. The cast is expected and harmless, so silence that message.
# Depending on the bitsandbytes version it arrives via `warnings` or via `logging`,
# so filter both.
_NOISE = "MatMul8bitLt: inputs will be cast"
warnings.filterwarnings("ignore", message=r".*MatMul8bitLt: inputs will be cast.*")


class _DropNoise(logging.Filter):
    def filter(self, record):
        return _NOISE not in record.getMessage()


logging.getLogger().addFilter(_DropNoise())
for _name in ("bitsandbytes", "bitsandbytes.autograd._functions", "py.warnings"):
    logging.getLogger(_name).addFilter(_DropNoise())

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Delimiters used in the prompt. "### Input:" also acts as the stop signal:
# a base model just continues the pattern, so once it starts a new example we cut it off.
IN_TAG = "### Input:"
OUT_TAG = "### Output:"


def load_jsonl_or_txt(path: Path, need_output: bool):
    items = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                obj = json.loads(line)
                if "input" not in obj or (need_output and "output" not in obj):
                    sys.exit(f"{path}:{n} must have 'input'" + (" and 'output'" if need_output else ""))
                items.append(obj)
            else:
                if need_output:
                    sys.exit("Examples must be a .jsonl file with 'input' and 'output' fields.")
                items.append({"input": line})
    return items


def build_prefix(examples, instruction):
    parts = []
    if instruction:
        parts.append(instruction.strip() + "\n\n")
    for ex in examples:
        parts.append(f"{IN_TAG}\n{ex['input'].strip()}\n{OUT_TAG}\n{ex['output'].strip()}\n\n")
    return "".join(parts)


def check_gpu():
    if not torch.cuda.is_available():
        sys.exit(
            "PyTorch can't see a CUDA GPU, so the model would load entirely on the CPU.\n"
            f"  torch version: {torch.__version__}  (a '+cpu' suffix means a CPU-only build)\n"
            "Fix: pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} | {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")


def load_model(model_id, four_bit, gpu_gib, cpu_gib):
    if four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        # Allow anything that doesn't fit on the GPU to sit in system RAM instead of crashing.
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

    kwargs = dict(
        quantization_config=quant,
        device_map="auto",
        max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"},
        dtype=torch.bfloat16,
    )

    # Qwen3.5 checkpoints are vision-language models. Older transformers versions may not
    # map them to the text-only class; only in that case fall back to the image-text class.
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise  # a real error (memory, quantization...) - show it instead of hiding it
        print("Text-only model class not supported for this checkpoint; using AutoModelForImageTextToText.")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    # Report where the weights ended up.
    devmap = getattr(model, "hf_device_map", {}) or {}
    offloaded = [k for k, v in devmap.items() if v in ("cpu", "disk")]
    if offloaded:
        print(f"Note: {len(offloaded)} module(s) offloaded to CPU RAM; expect slower runs. "
              "Try --4bit to keep everything on the GPU.")
    else:
        print("All weights are on the GPU.")

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # required for batched generation with decoder-only models
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def clean_output(text):
    # Stop at the start of the next example (or any stray tag), then trim.
    for stop in (IN_TAG, OUT_TAG):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("results.jsonl"), type=Path)
    ap.add_argument("--instruction", default="")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="use 4-bit instead of 8-bit")
    ap.add_argument("--gpu-gib", type=float, default=14, help="max GPU memory for weights (leave room for the prompt)")
    ap.add_argument("--cpu-gib", type=float, default=48, help="max system RAM for offloaded weights")
    args = ap.parse_args()

    examples = load_jsonl_or_txt(args.examples, need_output=True)
    data = load_jsonl_or_txt(args.data, need_output=False)
    prefix = build_prefix(examples, args.instruction)
    print(f"{len(examples)} examples, {len(data)} items to transform.")

    check_gpu()
    model, tok = load_model(args.model, args.four_bit, args.gpu_gib, args.cpu_gib)
    print(f"Prompt prefix is {len(tok(prefix).input_ids)} tokens.")

    # Libraries can reset warning filters while loading, so re-apply ours right before generating.
    warnings.filterwarnings("ignore", message=r".*MatMul8bitLt: inputs will be cast.*")
    # Logger-level filters don't catch child loggers, so also filter at the output handlers.
    for h in logging.getLogger().handlers + [logging.lastResort]:
        if h is not None:
            h.addFilter(_DropNoise())

    # Resume support: skip items already written to the output file.
    done = 0
    if args.out.exists():
        done = sum(1 for l in args.out.open(encoding="utf-8") if l.strip())
        if done:
            print(f"Resuming: {done} items already in {args.out}.")

    with args.out.open("a", encoding="utf-8") as fout:
        for start in range(done, len(data), args.batch_size):
            batch = data[start : start + args.batch_size]
            prompts = [f"{prefix}{IN_TAG}\n{item['input'].strip()}\n{OUT_TAG}\n" for item in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # deterministic: same input -> same output
                    stop_strings=[IN_TAG],
                    tokenizer=tok,
                    pad_token_id=tok.pad_token_id,
                )

            new_tokens = gen[:, enc.input_ids.shape[1] :]
            for item, ids in zip(batch, new_tokens):
                out = clean_output(tok.decode(ids, skip_special_tokens=True))
                fout.write(json.dumps({"input": item["input"], "output": out}, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  {min(start + args.batch_size, len(data))}/{len(data)}")

    print(f"Done. Results in {args.out}")


if __name__ == "__main__":
    main()