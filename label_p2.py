"""
P2 (prompt strategy) labeling for LiteLMGuard 2.0 using two models:
Claude Sonnet 5.5 (Anthropic Message Batches API) and GPT-5.6 Terra (OpenAI Batch API).

Usage:
  python label_p2.py list-models                  # confirm exact model IDs
  python label_p2.py submit --pilot 100           # pilot run on 100 random prompts
  python label_p2.py submit                       # full run
  python label_p2.py status                       # check both batches
  python label_p2.py collect                      # download + parse results
  python label_p2.py merge                        # agreement, disagreements, human sample

Input: the Answerable-or-Not dataset from Hugging Face (needs `pip install datasets`),
or a local CSV with the same columns. Column names are set in CONFIG below.
Env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd

# ---------------- CONFIG ----------------
# Hugging Face dataset ID, or a local CSV path
INPUT_SOURCE = "kalyannakka/Answerable-or-Not"
COL_ID = None                     # dataset has no ID column -> row index (p00000...)
COL_PROMPT = "question"
COL_ANSWERABLE = "answerable"     # YES / NO
COL_CATEGORY = "specific_harms"   # 61 fine-grained categories (P1 topic)

PROMPT_FILE = "p2_labeling_prompt.txt"
OUT_DIR = Path("p2_run")

CLAUDE_MODEL = "claude-sonnet-5-5"
OPENAI_MODEL = "gpt-5.6-terra"   # confirm with: python label_p2.py list-models

MAX_TOKENS_CLAUDE = 300
MAX_COMPLETION_TOKENS_OPENAI = 2000   # includes any reasoning tokens
OPENAI_CHUNK_SIZE = 400               # ~1.6k input tokens/request -> ~650k, under the 900k enqueued limit
HUMAN_SAMPLE_SIZE = 250
SEED = 42
# ----------------------------------------

LABELS = [
    "Plain Request",
    "Contextualized/Educational Framing",
    "Roleplay",
    "Hypothetical",
    "Indirect/Multistep",
    "Obfuscated",
    "Instruction Override",
    "Adversarial Attack",
    "Prompt Injection",
]
LABEL_LOOKUP = {l.lower(): l for l in LABELS}


def load_data(pilot=None):
    if INPUT_SOURCE.endswith(".csv"):
        df = pd.read_csv(INPUT_SOURCE)
    else:
        from datasets import load_dataset
        df = load_dataset(INPUT_SOURCE, split="train").to_pandas()
    if COL_ID is None or COL_ID not in df.columns:
        df["id"] = [f"p{i:05d}" for i in range(len(df))]
    else:
        df["id"] = df[COL_ID].astype(str)
    if pilot:
        df = df.sample(n=min(pilot, len(df)), random_state=SEED)
    return df


def user_message(prompt_text):
    return (
        "Label the prompt strategy of the following prompt.\n\n"
        f"<prompt_to_classify>\n{prompt_text}\n</prompt_to_classify>"
    )


def state_path():
    OUT_DIR.mkdir(exist_ok=True)
    return OUT_DIR / "state.json"


def save_state(state):
    state_path().write_text(json.dumps(state, indent=2))


def load_state():
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------- submit ----------------
def submit_claude(df, system_prompt):
    import anthropic
    client = anthropic.Anthropic()
    requests = []
    for _, row in df.iterrows():
        requests.append({
            "custom_id": row["id"],
            "params": {
                "model": CLAUDE_MODEL,
                "max_tokens": MAX_TOKENS_CLAUDE,
                # no temperature: deprecated for claude-sonnet-5-5 (request is rejected)
                # codebook is identical in every request -> cached
                "system": [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": [{"role": "user", "content": user_message(row[COL_PROMPT])}],
            },
        })
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def write_openai_jsonl(df, system_prompt, path):
    with path.open("w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "custom_id": row["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": OPENAI_MODEL,
                    "max_completion_tokens": MAX_COMPLETION_TOKENS_OPENAI,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message(row[COL_PROMPT])},
                    ],
                },
            }) + "\n")


def run_openai_chunked(df, system_prompt):
    """Submit OpenAI requests in chunks that fit the enqueued-token limit, one at a time.
    Completed chunk batch IDs are saved to state['openai_batches'] after each chunk,
    so an interrupted run can continue with `resume-openai`."""
    from openai import OpenAI
    client = OpenAI()
    state = load_state()
    done = state.get("openai_batches", [])
    chunks = [df.iloc[i:i + OPENAI_CHUNK_SIZE] for i in range(0, len(df), OPENAI_CHUNK_SIZE)]
    print(f"OpenAI: {len(chunks)} chunks of up to {OPENAI_CHUNK_SIZE}; {len(done)} already done")
    for idx, part in enumerate(chunks):
        if idx < len(done):
            continue
        while True:
            path = OUT_DIR / f"openai_requests_{idx:02d}.jsonl"
            write_openai_jsonl(part, system_prompt, path)
            uploaded = client.files.create(file=path.open("rb"), purpose="batch")
            batch = client.batches.create(input_file_id=uploaded.id,
                                          endpoint="/v1/chat/completions",
                                          completion_window="24h")
            print(f"  chunk {idx + 1}/{len(chunks)}: {batch.id}")
            while True:
                b = client.batches.retrieve(batch.id)
                if b.status in OPENAI_DONE:
                    break
                time.sleep(30)
            if b.status == "completed":
                done.append(batch.id)
                save_state({**load_state(), "openai_batches": done})
                print(f"  chunk {idx + 1} completed: {b.request_counts}")
                break
            codes = [e.code for e in (b.errors.data if b.errors else [])]
            if "token_limit_exceeded" in codes:
                print("  enqueued-token limit hit, retrying in 2 minutes")
                time.sleep(120)
                continue
            raise RuntimeError(f"Chunk {idx + 1} ended with status {b.status}: {b.errors}")
    save_state({**load_state(), "openai_complete": True})
    print("OpenAI: all chunks completed")


def openai_ids(state):
    if "openai_batches" in state:
        return state["openai_batches"]
    return [state["openai_batch"]] if "openai_batch" in state else []


def cmd_submit(args):
    df = load_data(args.pilot)
    system_prompt = Path(PROMPT_FILE).read_text()
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "input_used.csv", index=False)
    state = {**load_state(),
             "n": len(df),
             "pilot": args.pilot,
             "claude_model": CLAUDE_MODEL,
             "openai_model": OPENAI_MODEL,
             "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.only or args.only == "claude":
        state["claude_batch"] = submit_claude(df, system_prompt)
        print("Claude batch:", state["claude_batch"])
    if not args.only or args.only == "openai":
        state.pop("openai_batch", None)
        state["openai_batches"] = []
        state["openai_complete"] = False
    save_state(state)
    if not args.only or args.only == "openai":
        run_openai_chunked(df, system_prompt)


def cmd_resume_openai(_):
    df = pd.read_csv(OUT_DIR / "input_used.csv", dtype={"id": str})
    run_openai_chunked(df, Path(PROMPT_FILE).read_text())


# ---------------- status ----------------
def cmd_status(_):
    state = load_state()
    if "claude_batch" in state:
        import anthropic
        b = anthropic.Anthropic().messages.batches.retrieve(state["claude_batch"])
        print("Claude:", b.processing_status, b.request_counts)
    ids = openai_ids(state)
    if ids:
        from openai import OpenAI
        client = OpenAI()
        for i, bid in enumerate(ids):
            b = client.batches.retrieve(bid)
            print(f"OpenAI chunk {i + 1}:", b.status, b.request_counts)


# ---------------- parse helpers ----------------
def parse_label_json(text):
    """Return dict with strategy/secondary/confidence/rationale, or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    strat = LABEL_LOOKUP.get(str(obj.get("strategy", "")).strip().lower())
    if strat is None:
        return None
    sec = obj.get("secondary_strategy")
    sec = LABEL_LOOKUP.get(str(sec).strip().lower()) if sec else None
    return {
        "strategy": strat,
        "secondary": sec,
        "confidence": obj.get("confidence"),
        "rationale": obj.get("rationale"),
    }


# ---------------- collect ----------------
def collect_claude(batch_id):
    import anthropic
    client = anthropic.Anthropic()
    rows = []
    for entry in client.messages.batches.results(batch_id):
        rec = {"id": entry.custom_id, "raw": None, "error": None}
        if entry.result.type == "succeeded":
            msg = entry.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            rec["raw"] = text or None
            if not text:
                rec["error"] = f"empty response (stop_reason={msg.stop_reason})"
        else:
            err = getattr(entry.result, "error", None)
            detail = getattr(getattr(err, "error", None), "message", None) or str(err)
            rec["error"] = f"{entry.result.type}: {detail}"
        rows.append(rec)
    return rows


def collect_openai(batch_id):
    from openai import OpenAI
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    rows = []
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            obj = json.loads(line)
            rec = {"id": obj["custom_id"], "raw": None, "error": None}
            resp = obj.get("response") or {}
            if resp.get("status_code") == 200:
                rec["raw"] = resp["body"]["choices"][0]["message"].get("content")
            else:
                rec["error"] = str(obj.get("error") or resp.get("status_code"))
            rows.append(rec)
    if b.error_file_id:
        for line in client.files.content(b.error_file_id).text.splitlines():
            obj = json.loads(line)
            rows.append({"id": obj["custom_id"], "raw": None, "error": str(obj.get("error"))})
    return rows


def to_frame(rows, prefix):
    out = []
    for r in rows:
        parsed = parse_label_json(r["raw"])
        out.append({
            "id": r["id"],
            f"{prefix}_strategy": parsed["strategy"] if parsed else None,
            f"{prefix}_secondary": parsed["secondary"] if parsed else None,
            f"{prefix}_confidence": parsed["confidence"] if parsed else None,
            f"{prefix}_rationale": parsed["rationale"] if parsed else None,
            f"{prefix}_error": r["error"] or (None if parsed else "unparseable"),
            f"{prefix}_raw": r["raw"],
        })
    return pd.DataFrame(out)


OPENAI_DONE = {"completed", "failed", "expired", "cancelled"}


def batch_states(state):
    out = {}
    if "claude_batch" in state:
        import anthropic
        out["claude"] = anthropic.Anthropic().messages.batches.retrieve(state["claude_batch"]).processing_status
    if "openai_batches" in state:
        out["openai"] = "completed" if state.get("openai_complete") else "chunks_running"
    elif "openai_batch" in state:
        from openai import OpenAI
        out["openai"] = OpenAI().batches.retrieve(state["openai_batch"]).status
    return out


def cmd_collect(args):
    state = load_state()
    if getattr(args, "wait", False):
        while True:
            state = load_state()
            s = batch_states(state)
            done = (s.get("claude", "ended") == "ended") and (s.get("openai", "completed") in OPENAI_DONE)
            print(time.strftime("%H:%M:%S"), s)
            if done:
                break
            time.sleep(60)
    s = batch_states(state)
    if "claude_batch" in state:
        if s["claude"] != "ended":
            print(f"Claude: not finished yet ({s['claude']}), skipping")
        else:
            df = to_frame(collect_claude(state["claude_batch"]), "claude")
            df.to_csv(OUT_DIR / "claude_labels.csv", index=False)
            print(f"Claude: {df['claude_strategy'].notna().sum()}/{len(df)} parsed")
            errs = df["claude_error"].dropna()
            if len(errs):
                print(f"  {len(errs)} errors, first: {errs.iloc[0]}")
    if openai_ids(state):
        if s["openai"] not in OPENAI_DONE:
            print(f"OpenAI: not finished yet ({s['openai']}), skipping")
        else:
            rows = [r for bid in openai_ids(state) for r in collect_openai(bid)]
            df = to_frame(rows, "gpt")
            df.to_csv(OUT_DIR / "gpt_labels.csv", index=False)
            print(f"GPT: {df['gpt_strategy'].notna().sum()}/{len(df)} parsed (batch status: {s['openai']})")


# ---------------- merge ----------------
def final_p2(strategy, answerable):
    """Plain Request splits into Benign / Harmful Direct Request using the v1 label."""
    if strategy != "Plain Request":
        return strategy
    return "Benign" if str(answerable).strip().upper() == "YES" else "Harmful Direct Request"


def cmd_merge(_):
    data = pd.read_csv(OUT_DIR / "input_used.csv", dtype={"id": str})
    c = pd.read_csv(OUT_DIR / "claude_labels.csv", dtype={"id": str})
    g = pd.read_csv(OUT_DIR / "gpt_labels.csv", dtype={"id": str})
    df = data.merge(c, on="id", how="left").merge(g, on="id", how="left")

    both = df["claude_strategy"].notna() & df["gpt_strategy"].notna()
    df["agree"] = both & (df["claude_strategy"] == df["gpt_strategy"])
    df["status"] = "disagree"
    df.loc[df["agree"], "status"] = "agreed"
    df.loc[~both, "status"] = "missing_label"

    df["p2_final"] = None
    df.loc[df["agree"], "p2_final"] = [
        final_p2(s, a) for s, a in zip(df.loc[df["agree"], "claude_strategy"],
                                       df.loc[df["agree"], COL_ANSWERABLE])
    ]

    df.to_csv(OUT_DIR / "merged.csv", index=False)
    df[df["status"] != "agreed"].to_csv(OUT_DIR / "to_adjudicate.csv", index=False)

    # Human kappa sample: stratified by P1 category, labels hidden
    n_per = max(1, HUMAN_SAMPLE_SIZE // df[COL_CATEGORY].nunique())
    sample = pd.concat([
        g.sample(n=min(n_per, len(g)), random_state=SEED)
        for _, g in df.groupby(COL_CATEGORY)
    ])
    sample = sample.sample(frac=1, random_state=SEED)
    sample[["id", COL_PROMPT]].assign(human_strategy="").to_csv(
        OUT_DIR / "human_sample_blind.csv", index=False)

    print(f"Total: {len(df)}")
    print(f"Agreed: {df['agree'].sum()} ({df['agree'].mean():.1%})")
    print(f"Disagree: {(df['status']=='disagree').sum()}   Missing: {(df['status']=='missing_label').sum()}")
    print("\nFinal P2 distribution (agreed only):")
    print(df["p2_final"].value_counts())
    print("\nConfusion (Claude rows x GPT cols):")
    print(pd.crosstab(df["claude_strategy"], df["gpt_strategy"]))
    print(f"\nWrote merged.csv, to_adjudicate.csv, human_sample_blind.csv ({len(sample)} rows) to {OUT_DIR}/")


def cmd_kappa(_):
    """After filling human_strategy in human_sample_blind.csv."""
    from sklearn.metrics import cohen_kappa_score
    h = pd.read_csv(OUT_DIR / "human_sample_blind.csv", dtype={"id": str})
    m = pd.read_csv(OUT_DIR / "merged.csv", dtype={"id": str})
    df = h.merge(m, on="id")
    df = df[df["human_strategy"].notna() & (df["human_strategy"] != "")]
    for col in ["claude_strategy", "gpt_strategy"]:
        sub = df[df[col].notna()]
        print(f"kappa human vs {col}: {cohen_kappa_score(sub['human_strategy'], sub[col]):.3f} (n={len(sub)})")


def cmd_list_models(_):
    try:
        from openai import OpenAI
        print("OpenAI:", sorted(m.id for m in OpenAI().models.list() if "gpt-5" in m.id))
    except Exception as e:
        print("OpenAI: failed ->", e)
    try:
        import anthropic
        print("Anthropic:", [m.id for m in anthropic.Anthropic().models.list()])
    except Exception as e:
        print("Anthropic: failed ->", e)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--pilot", type=int, default=None)
    s.add_argument("--only", choices=["claude", "openai"], default=None)
    sub.add_parser("status")
    sub.add_parser("resume-openai")
    c = sub.add_parser("collect")
    c.add_argument("--wait", action="store_true", help="poll every 60s until both batches finish")
    sub.add_parser("merge")
    sub.add_parser("kappa")
    sub.add_parser("list-models")
    args = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect,
     "resume-openai": cmd_resume_openai,
     "merge": cmd_merge, "kappa": cmd_kappa, "list-models": cmd_list_models}[args.cmd](args)


if __name__ == "__main__":
    main()
