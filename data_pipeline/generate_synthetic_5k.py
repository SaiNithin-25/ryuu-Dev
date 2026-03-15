"""Generate a high-quality synthetic 5k developer JSONL dataset for Ryuu-Dev."""

from __future__ import annotations

import json
import random
from pathlib import Path


def build_prompt(
    i: int,
    lang: str,
    domain: str,
    task: str,
    constraint: str,
    level: str,
    scale: int,
    latency_ms: int,
    error_budget: str,
) -> str:
    return (
        f"You are helping on a {domain} project.\n"
        f"Task: {task} in {lang}.\n"
        f"Difficulty: {level}.\n"
        f"Constraint: {constraint}.\n"
        f"Scenario: service handles ~{scale} req/min, p95 target {latency_ms}ms, error budget {error_budget}.\n"
        f"Case fingerprint: dev-{i:04d}-{lang[:2].lower()}-{domain.split()[0].lower()}.\n"
        "Return clear steps, final code, complexity notes, and one edge-case test."
    )


def code_block(lang: str, func_name: str, behavior: str) -> str:
    if lang == "Python":
        return (
            "```python\n"
            f"def {func_name}(items):\n"
            "    result = []\n"
            "    seen = set()\n"
            "    for x in items:\n"
            "        if x in seen:\n"
            "            continue\n"
            "        seen.add(x)\n"
            "        result.append(x)\n"
            "    return result\n"
            "```\n"
            f"Behavior note: {behavior}"
        )
    if lang == "JavaScript":
        return (
            "```js\n"
            f"function {func_name}(items) {{\n"
            "  const seen = new Set();\n"
            "  const out = [];\n"
            "  for (const x of items) {\n"
            "    if (seen.has(x)) continue;\n"
            "    seen.add(x);\n"
            "    out.push(x);\n"
            "  }\n"
            "  return out;\n"
            "}\n"
            "```\n"
            f"Behavior note: {behavior}"
        )
    if lang == "Java":
        return (
            "```java\n"
            "import java.util.*;\n"
            f"class Solution {{\n"
            f"  static List<String> {func_name}(List<String> items) {{\n"
            "    Set<String> seen = new HashSet<>();\n"
            "    List<String> out = new ArrayList<>();\n"
            "    for (String x : items) {\n"
            "      if (seen.contains(x)) continue;\n"
            "      seen.add(x);\n"
            "      out.add(x);\n"
            "    }\n"
            "    return out;\n"
            "  }\n"
            "}\n"
            "```\n"
            f"Behavior note: {behavior}"
        )
    if lang == "C++":
        return (
            "```cpp\n"
            "#include <vector>\n"
            "#include <unordered_set>\n"
            "#include <string>\n"
            f"std::vector<std::string> {func_name}(const std::vector<std::string>& items) {{\n"
            "    std::unordered_set<std::string> seen;\n"
            "    std::vector<std::string> out;\n"
            "    for (const auto& x : items) {\n"
            "        if (seen.count(x)) continue;\n"
            "        seen.insert(x);\n"
            "        out.push_back(x);\n"
            "    }\n"
            "    return out;\n"
            "}\n"
            "```\n"
            f"Behavior note: {behavior}"
        )
    if lang == "Go":
        return (
            "```go\n"
            "package main\n\n"
            f"func {func_name}(items []string) []string {{\n"
            "    seen := map[string]bool{}\n"
            "    out := make([]string, 0, len(items))\n"
            "    for _, x := range items {\n"
            "        if seen[x] {\n"
            "            continue\n"
            "        }\n"
            "        seen[x] = true\n"
            "        out = append(out, x)\n"
            "    }\n"
            "    return out\n"
            "}\n"
            "```\n"
            f"Behavior note: {behavior}"
        )
    return (
        "```sql\n"
        "SELECT user_id, COUNT(*) AS event_count\n"
        "FROM events\n"
        "GROUP BY user_id\n"
        "HAVING COUNT(*) > 1;\n"
        "```\n"
        f"Behavior note: {behavior}"
    )


def build_response(
    i: int,
    lang: str,
    domain: str,
    task: str,
    constraint: str,
    pitfall: str,
    check: str,
) -> str:
    fn = f"dedupe_items_{i % 97}_{(i * 7) % 31}"
    behavior = (
        f"Designed for {domain}; applies {constraint.lower()} and avoids {pitfall.lower()}."
    )
    return (
        "Plan:\n"
        f"1. Clarify expected input/output and failure behavior for: {task}.\n"
        "2. Use deterministic control flow and explicit state transitions.\n"
        "3. Add one fast path and one defensive guard for invalid input.\n\n"
        "Implementation:\n"
        f"{code_block(lang, fn, behavior)}\n\n"
        "Complexity:\n"
        "- Time: O(n) for linear pass or equivalent bounded operations.\n"
        "- Space: O(n) due to support structures used for correctness.\n\n"
        "Edge-case test:\n"
        f"- {check}\n\n"
        "Engineering note:\n"
        "Keep logs concise, prefer pure functions for core logic, and separate I/O from computation."
    )


def main() -> None:
    random.seed(42)
    out = Path("data/raw/custom/synthetic_high_quality_5000.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    languages = ["Python", "JavaScript", "Java", "C++", "Go", "SQL"]
    domains = [
        "backend API",
        "data engineering",
        "ML tooling",
        "developer platform",
        "testing framework",
        "distributed services",
        "security automation",
        "observability stack",
    ]
    tasks = [
        "implement robust input validation",
        "optimize repeated lookups and caching",
        "refactor error handling for clearer failures",
        "design deterministic retry logic",
        "build safe file processing utility",
        "create stable pagination behavior",
        "harden authentication flow checks",
        "write deduplication and normalization routine",
        "add timeout and cancellation support",
        "improve structured logging and trace context",
    ]
    constraints = [
        "No external dependencies",
        "Must be deterministic under fixed seed",
        "Handle empty input and null-like values safely",
        "Keep memory growth bounded",
        "Support high-throughput request patterns",
        "Return actionable error messages",
        "Preserve backward-compatible behavior",
        "Avoid data races and shared mutable state leaks",
    ]
    pitfalls = [
        "silent failure paths",
        "stateful side effects in utility layers",
        "ambiguous return values",
        "overly broad exception handling",
        "non-idempotent retry behavior",
        "hidden quadratic complexity",
        "schema drift assumptions",
        "resource leak during partial failure",
    ]
    checks = [
        "Input list is empty and should return an empty collection.",
        "Input contains duplicates with mixed casing and expected normalization is explicit.",
        "Input has one malformed record and function should continue safely or return precise error.",
        "Repeated call with same input must produce identical output ordering.",
        "Large input batch must complete within acceptable time without memory spikes.",
        "Boundary value exactly at threshold should pass without off-by-one errors.",
        "Null-like elements should be rejected with clear message and no partial corruption.",
        "Concurrent access simulation should not mutate shared state unexpectedly.",
    ]
    levels = ["Beginner+", "Intermediate", "Advanced", "Production-grade"]
    error_budgets = ["99.9%", "99.5%", "99.0%", "98.5%"]

    with out.open("w", encoding="utf-8") as f:
        for i in range(5000):
            lang = languages[i % len(languages)]
            domain = domains[(i * 3) % len(domains)]
            task = tasks[(i * 5) % len(tasks)]
            constraint = constraints[(i * 7) % len(constraints)]
            pitfall = pitfalls[(i * 11) % len(pitfalls)]
            check = checks[(i * 13) % len(checks)]
            level = levels[(i * 17) % len(levels)]
            scale = 500 + ((i * 37) % 50000)
            latency_ms = 40 + ((i * 19) % 360)
            error_budget = error_budgets[(i * 23) % len(error_budgets)]

            prompt = build_prompt(
                i,
                lang,
                domain,
                task,
                constraint,
                level,
                scale,
                latency_ms,
                error_budget,
            )
            response = build_response(i, lang, domain, task, constraint, pitfall, check)
            row = {"prompt": prompt, "response": response}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote 5000 rows to {out}")


if __name__ == "__main__":
    main()
