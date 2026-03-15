"""Download selected coding datasets and export to data/raw/hf/*.jsonl.

Usage:
  cuda\Scripts\python.exe data_pipeline/10_download_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


TARGETS = [
    {
        "name": "taco",
        "hf_id": "BAAI/TACO",
        "split": "train",
        "limit": 50000,
    },
    {
        "name": "apps",
        "hf_id": "codeparrot/apps",
        "split": "train",
        "limit": 50000,
    },
    {
        "name": "code_contests",
        "hf_id": "deepmind/code_contests",
        "split": "train",
        "limit": 30000,
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_record(ds_name: str, row: Dict) -> Dict | None:
    # Common fields across coding datasets.
    if ds_name == "taco":
        prompt = row.get("question") or row.get("prompt")
        sols = row.get("solutions")
        response = sols[0] if isinstance(sols, list) and sols else row.get("answer")
        if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
            return {"prompt": prompt.strip(), "response": response.strip()}
        return None

    if ds_name == "apps":
        prompt = row.get("question")
        sols = row.get("solutions")
        response = None
        if isinstance(sols, str) and sols.strip():
            # APPS may store solutions as JSON string.
            try:
                parsed = json.loads(sols)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                    response = parsed[0]
            except Exception:
                response = sols
        elif isinstance(sols, list) and sols and isinstance(sols[0], str):
            response = sols[0]
        if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
            return {"prompt": prompt.strip(), "response": response.strip()}
        return None

    if ds_name == "code_contests":
        desc = row.get("description")
        sols = row.get("solutions")
        response = None
        if isinstance(sols, dict):
            lang = sols.get("language")
            code = sols.get("solution")
            if isinstance(code, list) and code and isinstance(code[0], str):
                response = code[0]
            elif isinstance(code, str):
                response = code
            if isinstance(lang, list) and lang:
                if response:
                    response = f"Language: {lang[0]}\n\n{response}"
        if isinstance(desc, str) and isinstance(response, str) and desc.strip() and response.strip():
            return {"prompt": desc.strip(), "response": response.strip()}
        return None

    return None


def normalize_generic(row: Dict) -> Dict | None:
    prompt_keys = ["prompt", "instruction", "question", "problem", "query", "input"]
    response_keys = ["response", "output", "answer", "solution", "final", "assistant"]

    prompt = ""
    response = ""

    for k in prompt_keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            prompt = v.strip()
            break

    for k in response_keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            response = v.strip()
            break

    if not prompt or not response:
        return None
    return {"prompt": prompt, "response": response}


def download_one(target: Dict) -> int:
    from datasets import load_dataset

    ds_name = target["name"]
    hf_id = target["hf_id"]
    split = target["split"]
    limit = int(target["limit"])

    out_dir = Path("data/raw/hf") / ds_name
    ensure_dir(out_dir)
    out_path = out_dir / f"{split}.jsonl"

    try:
        ds = load_dataset(hf_id, split=split)
    except Exception as e:
        msg = str(e).lower()
        # datasets>=4 removed script-based loaders; use parquet API fallback.
        if "dataset scripts are no longer supported" in msg or "loading script" in msg:
            print(f"[WARN] {ds_name}: script-based loader blocked, trying parquet API fallback...")
            ds = load_dataset_from_parquet_api(hf_id, split)
        else:
            raise
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            rec = normalize_record(ds_name, row)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= limit:
                break
    print(f"[OK] {ds_name}: wrote {kept} rows -> {out_path}")
    return kept


def load_dataset_from_parquet_api(hf_id: str, split: str):
    import requests
    from datasets import load_dataset

    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    files = payload.get("parquet_files", [])
    urls = [f["url"] for f in files if f.get("split") == split and isinstance(f.get("url"), str)]
    if not urls:
        # Fallback: use all splits if split key missing.
        urls = [f["url"] for f in files if isinstance(f.get("url"), str)]
    if not urls:
        raise RuntimeError(f"No parquet URLs found for {hf_id} ({split})")
    return load_dataset("parquet", data_files=urls, split="train")


def main() -> int:
    try:
        import datasets  # noqa: F401
    except Exception:
        print("[ERR] Missing dependency: datasets")
        print("Install with: cuda\\Scripts\\python.exe -m pip install datasets")
        return 1

    total = 0
    for target in TARGETS:
        try:
            total += download_one(target)
        except Exception as e:
            print(f"[ERR] Failed {target['name']}: {e}")
    print(f"[DONE] Total rows exported: {total}")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
