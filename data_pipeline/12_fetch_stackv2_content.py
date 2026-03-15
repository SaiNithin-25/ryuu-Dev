"""Fetch Stack v2 code content from Software Heritage S3 using blob_id.

Requirements:
  - AWS credentials in environment:
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  - Optional: AWS_SESSION_TOKEN for temporary creds
  - pip install smart_open[s3] boto3 pyarrow

Input:
  - Parquet shards with files[*].blob_id, path, language, length_bytes, license_type, is_vendor, is_generated
Output:
  - JSONL with prompt/response pairs for training
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Dict, List

import boto3
import pyarrow.parquet as pq
from smart_open import open as s3_open

SAFE_LANGS = {
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go", "Rust",
    "SQL", "Shell", "Bash", "Dockerfile", "HTML", "CSS", "JSON", "YAML", "TOML",
    "Markdown", "Makefile"
}


def iter_parquet_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.parquet"))


def iter_rows(parquet_path: Path):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=1000):
        yield batch.to_pydict()


def download_blob(s3_client, blob_id: str) -> bytes | None:
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    try:
        with s3_open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as fin:
            return fin.read()
    except Exception:
        return None


def to_prompt_response(path: str, content: str) -> Dict[str, str]:
    prompt = f"Explain and summarize the following code file: {path}"
    response = content
    return {"prompt": prompt, "response": response}


def main() -> int:
    input_dir = Path("data/raw/hf/the-stack-v2-train-smol-ids")
    output_path = Path("data/raw/hf/stackv2_content/train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not aws_key or not aws_secret:
        print("[ERR] Missing AWS credentials in environment")
        return 1

    session = boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    s3 = session.client("s3")

    total_files = 0
    kept_files = 0

    with output_path.open("w", encoding="utf-8") as out:
        for parquet_file in iter_parquet_files(input_dir):
            for batch in iter_rows(parquet_file):
                files_list = batch.get("files")
                if not files_list:
                    continue

                # files_list is list of lists of structs (per repo row)
                for files in files_list:
                    if not files:
                        continue
                    for f in files:
                        total_files += 1
                        lang = f.get("language")
                        length = f.get("length_bytes") or 0
                        is_vendor = f.get("is_vendor")
                        is_generated = f.get("is_generated")

                        if lang not in SAFE_LANGS:
                            continue
                        if is_vendor or is_generated:
                            continue
                        if length < 200 or length > 200_000:
                            continue

                        blob_id = f.get("blob_id")
                        path = f.get("path", "unknown")
                        if not blob_id:
                            continue

                        content_bytes = download_blob(s3, blob_id)
                        if not content_bytes:
                            continue

                        try:
                            content = content_bytes.decode(f.get("src_encoding") or "utf-8", errors="ignore")
                        except Exception:
                            content = content_bytes.decode("utf-8", errors="ignore")

                        content = content.strip()
                        if not content:
                            continue

                        rec = to_prompt_response(path, content)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept_files += 1

                        if kept_files % 500 == 0:
                            print(f"[OK] Kept {kept_files} / scanned {total_files}")

    print(f"[DONE] Wrote {kept_files} files to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
