"""Apply quality filters focused for Ryuu-Dev coding assistant data."""

import json
from multiprocessing import Pool, cpu_count
from data_pipeline.common import (
    load_config,
    read_jsonl,
    basic_token_count,
    ascii_ratio,
    repeat_line_ratio,
    url_ratio,
    symbol_ratio,
    unique_words_ratio,
)

_FILTER_Q = None


def _init_filter(q):
    global _FILTER_Q
    _FILTER_Q = q


def _filter_worker(row):
    q = _FILTER_Q
    p = row["prompt"].strip()
    r = row["response"].strip()

    if not (q["min_prompt_chars"] <= len(p) <= q["max_prompt_chars"]):
        return ("prompt_len", None)
    if not (q["min_response_chars"] <= len(r) <= q["max_response_chars"]):
        return ("response_len", None)

    p_tok = basic_token_count(p)
    r_tok = basic_token_count(r)
    if p_tok < q["min_prompt_tokens"] or r_tok < q["min_response_tokens"] or (p_tok + r_tok) > q["max_total_tokens"]:
        return ("token_len", None)

    if ascii_ratio(r) < q["ascii_ratio_floor"]:
        return ("ascii_ratio", None)
    if repeat_line_ratio(r) > q["max_repeat_line_ratio"]:
        return ("repeat_lines", None)
    if url_ratio(r) > q["max_url_ratio"]:
        return ("url_ratio", None)
    if symbol_ratio(r) > q["max_symbol_ratio"]:
        return ("symbol_ratio", None)
    if unique_words_ratio(r) < q["min_unique_words_ratio"]:
        return ("unique_words", None)

    lowered = (p + "\n" + r).lower()
    if any(bad.lower() in lowered for bad in q.get("forbidden_substrings", [])):
        return ("forbidden_substrings", None)

    return ("keep", {
        "prompt": p,
        "response": r,
        "_source": row.get("_source", "unknown"),
        "_source_file": row.get("_source_file", ""),
    })


def main():
    cfg = load_config()
    q = cfg["quality"]

    in_path = cfg["output"]["cleaned"]
    out_path = cfg["output"]["filtered"]

    dropped = {
        "prompt_len": 0,
        "response_len": 0,
        "token_len": 0,
        "ascii_ratio": 0,
        "repeat_lines": 0,
        "url_ratio": 0,
        "symbol_ratio": 0,
        "unique_words": 0,
        "forbidden_substrings": 0,
    }

    kept = 0
    perf = cfg.get("performance", {})
    workers = max(1, int(perf.get("workers", 1)))
    chunk_size = max(100, int(perf.get("chunk_size", 2000)))
    with open(out_path, "w", encoding="utf-8") as out:
        if workers == 1:
            for row in read_jsonl(in_path):
                status, payload = _filter_worker(row)
                if status == "keep":
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    dropped[status] += 1
        else:
            pool = Pool(processes=min(workers, cpu_count()), initializer=_init_filter, initargs=(q,))
            try:
                buf = []
                for row in read_jsonl(in_path):
                    buf.append(row)
                    if len(buf) >= chunk_size:
                        results = pool.map(_filter_worker, buf)
                        for status, payload in results:
                            if status == "keep":
                                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                                kept += 1
                            else:
                                dropped[status] += 1
                        buf = []
                if buf:
                    results = pool.map(_filter_worker, buf)
                    for status, payload in results:
                        if status == "keep":
                            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            kept += 1
                        else:
                            dropped[status] += 1
            finally:
                pool.close()
                pool.join()

    print(f"[OK] Filtered rows: {kept}")
    print("[OK] Dropped breakdown:")
    for k, v in dropped.items():
        print(f"  - {k}: {v}")
    print(f"[OK] Output: {out_path}")


if __name__ == "__main__":
    main()
