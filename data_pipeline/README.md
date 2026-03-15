# Ryuu-Dev Data Pipeline

This pipeline rebuilds all training artifacts from raw data:

1. collect source files
2. normalize prompt/response pairs
3. quality filter
4. deduplicate
5. leakage-safe train/test split
6. tokenizer training
7. tokenization + shard building
8. DPO seed pair generation
9. validation

## Expected Raw Data Layout

Put datasets under:

- `data/raw/tinycodes/*.jsonl`
- `data/raw/taco/*.jsonl`
- `data/raw/custom/**/*.*`
- `data/raw/doc/**/*.*`
- `data/raw/hf/code_contests/train.jsonl`
- `data/raw/hf/apps/train.jsonl`
- `data/raw/hf/taco/train.jsonl` (produced by `11_convert_taco_arrow.py`)
- `data/raw/hf/openmathinstruct/correct_solutions/train .jsonl`
- `data/raw/hf/openmathinstruct/incorrect_solutions/train.jsonl`
- `data/raw/hf/ling_coder_sft/train.jsonl` (produced by `13_convert_ling_coder_sft.py`)
- `data/raw/hf/stackv2_content/train.jsonl` (from Stack v2 S3 content fetch)

Supported file formats:

- `.jsonl` or `.json`: one JSON object per line
- `.parquet`: optional (requires `pyarrow`)
- `.txt` / `.md`: wrapped as a summarize-task sample
- `.pdf`: extracted page text (requires `pypdf`)
- `.docx`: extracted paragraph text (`python-docx` preferred; zip/xml fallback included)
- `.doc`: skipped by default (convert to `.docx` or `.txt`)

Accepted schema is flexible. Normalizer supports:

- prompt-like keys: `prompt`, `instruction`, `question`, `input`, `query`, `task`
- response-like keys: `response`, `completion`, `output`, `answer`, `solution`, `code`
- chat-style: `conversations`/`messages` with user/assistant roles
- list fields: `solutions`, `answers` (first item used)

## Run Full Pipeline

Windows:

```bat
run_data_pipeline.bat
```

Direct Python:

```powershell
cuda\Scripts\python.exe data_pipeline\run_full_pipeline.py --python cuda\Scripts\python.exe
```

## Download Coding Datasets

Configured downloader script:

- `data_pipeline/10_download_datasets.py`

Current dataset targets:

- `BAAI/TACO` (`train`, exported to `data/raw/hf/taco/train.jsonl`)
- `codeparrot/apps` (`train`, exported to `data/raw/hf/apps/train.jsonl`)
- `deepmind/code_contests` (`train`, exported to `data/raw/hf/code_contests/train.jsonl`)

Run:

```bat
run_download_datasets.bat
```

Then rebuild artifacts:

```bat
run_data_pipeline.bat
```

## Ling-Coder conversion

SFT:
```bat
cuda\Scripts\python.exe data_pipeline\13_convert_ling_coder_sft.py
```

DPO:
```bat
cuda\Scripts\python.exe data_pipeline\14_convert_ling_coder_dpo.py
```

Sampling (1M cap for Ling-Coder-SFT):
```bat
run_sample_ling_coder_sft_1M.bat
```

Optional JSONL -> Parquet (speed up large JSONL reads):
```bat
cuda\Scripts\python.exe data_pipeline\16_jsonl_to_parquet.py --input <file.jsonl> --output <file.parquet>
```

Pipeline report:
- `data/open/Ryuu_Developer_v1/pipeline_report.json`

## Stack v2 Content Fetch (IDs -> code)

If you have Stack v2 ID shards (`the-stack-v2-train-smol-ids`), you must fetch content from S3:

Requirements:
- AWS credentials in env (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`)
- `pip install smart_open[s3] boto3 pyarrow`

Run:

```bat
cuda\Scripts\python.exe data_pipeline\12_fetch_stackv2_content.py
```

## Outputs

- `data/open/Ryuu_Developer_v1/train.jsonl`
- `data/open/Ryuu_Developer_v1/test.jsonl`
- `data/open/Ryuu_Developer_v1/stats.json`
- `tokenizer/bpe_tokenizer_postproc.json`
- `data/open/Ryuu_Developer_v1/tokenized/train_shard*.bin/.idx`
- `data/open/Ryuu_Developer_v1/tokenized/test_shard*.bin/.idx`
- `data/open/Ryuu_Developer_v1/tokenized/tokenization_stats.json`
- `data/dpo/ryuu_dpo_seed.jsonl`

## Quality Controls

Configured in `data_pipeline/pipeline_config.json`:

- char/token length limits
- ascii ratio
- repeated line ratio
- URL density cap
- symbol density cap
- unique-word ratio floor
- forbidden substring blocklist
- prompt-overlap threshold for train/test

## Validation Gate

`09_validate_pipeline.py` fails the run if:

- train/test rows below configured minimum
- schema violations in splits
- tokenizer missing
- shard files missing
- prompt overlap too high
- DPO schema broken
