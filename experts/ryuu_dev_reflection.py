# experts/ryuu_dev_reflection.py

import torch
import math
from typing import Dict, Any, Optional


class RyuuDevReflectionDriver:
    """
    Self-reflection driver for Ryuu-dev (inference only)

    Loop:
      1. Generate initial answer
      2. Inspect reasoning confidence
      3. If weak -> critique
      4. Revise answer
    """

    def __init__(
        self,
        model,
        tokenizer,
        confidence_threshold: float = 0.55,
        max_reflection_steps: int = 1,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.max_reflection_steps = max_reflection_steps
        self.temperature = temperature

    # --------------------------------------------------
    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 256):
        ids = self.tokenizer.encode_ids(prompt)
        x = torch.tensor(ids, device=self.model.device).unsqueeze(0)

        _, _, _, reasoning = self.model(
            x,
            targets=None,
        )
        if isinstance(reasoning, dict):
            reasoning["input_ids"] = x

        out = self.model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_k=40,
            do_sample=True,
            eos_token_id=self.tokenizer.token_to_id("</s>"),
        )
        text = self.tokenizer.decode(out[0].tolist())
        return text, reasoning

    # --------------------------------------------------
    def _estimate_confidence(self, reasoning: Optional[Dict[str, Any]]) -> float:
        """
        Convert reasoning diagnostics into scalar confidence [0,1]
        """

        if reasoning is None:
            return 0.0

        info = reasoning.get("info", {})
        scale = float(info.get("scale", 0.0))

        if scale <= 0.0:
            return 0.0  # curriculum off

        token_weights = reasoning.get("token_weights", None)
        if token_weights is None:
            return 0.3 * scale

        # Attention concentration = confidence
        entropy = -(token_weights * torch.log(token_weights + 1e-8)).sum(dim=-1)
        entropy = entropy.mean().item()

        max_entropy = math.log(token_weights.size(-1))
        attn_conf = 1.0 - min(entropy / max_entropy, 1.0)

        # Reasoning vector norm
        scores = reasoning.get("scores", None)
        if scores is not None:
            norm = torch.norm(scores, dim=-1).mean().item()
            norm_conf = min(norm / 5.0, 1.0)
        else:
            norm_conf = 0.5

        # Weighted blend
        confidence = (
            0.4 * attn_conf +
            0.4 * norm_conf +
            0.2 * scale
        )

        return float(max(0.0, min(1.0, confidence)))

    # --------------------------------------------------
    def _critique_prompt(self, answer: str) -> str:
        return (
            "You are Ryuu-dev. Critically review the following answer.\n"
            "Identify logical gaps, missing steps, or incorrect assumptions.\n\n"
            f"Answer:\n{answer}\n\n"
            "Critique:"
        )

    def _targeted_critique_prompt(self, weak_spans):
        spans = "\n".join(f"- {s}" for s in weak_spans)
        return f"""
        You are reviewing ONLY the following parts of an answer that show weak reasoning.
        Do NOT rewrite the full answer.
        Do NOT change correct parts.

        weak sections:
        {spans}
        Explain:
        - What is wrong
        - What is missing
        How to fix it concisely
        """

    def _revise_prompt(self, original: str, critique: str) -> str:
        return (
            "You are Ryuu-dev. Improve the answer using the critique below.\n"
            "Produce a corrected, clearer, and more complete solution.\n\n"
            f"Original Answer:\n{original}\n\n"
            f"Critique:\n{critique}\n\n"
            "Revised Answer:"
        )

    def _targeted_revision_prompt(self, original, critique):
        return f"""
        Original answer:
        {original}

        Revise ONLY the criticized parts.
        Keep everything else unchanged
        """

    # --------------------------------------------------
    def _extract_weak_spans(
        self,
        input_ids: torch.Tensor,
        token_weights: torch.Tensor,
        tokenizer,
        threshold: float = 0.15,
    ):
        """
        Returns text spans corresponding to weak reasoning tokens.
        """
        ids = input_ids[0].tolist()
        weights = token_weights[0].tolist()

        weak_ids = [
            tid for tid, w in zip(ids, weights) if w < threshold
        ]

        if not weak_ids:
            return []

        text = tokenizer.decode(weak_ids)
        return [text.strip()]

    # --------------------------------------------------
    @torch.no_grad()
    def run(self, prompt: str) -> Dict[str, Any]:
        self.model.eval()

        # 1) Initial generation
        answer, reasoning = self._generate(prompt)
        conf_before = self._estimate_confidence(reasoning)

        result = {
            "initial_answer": answer,
            "final_answer": answer,
            "confidence_before": conf_before,
            "confidence_after": conf_before,
            "confidence_delta": 0.0,
            "reflected": False,
        }

        # 2) Decide reflection
        if conf_before >= self.confidence_threshold:
            return result

        current_answer = answer
        current_conf = conf_before

        # 3) Reflection loop
        for _ in range(self.max_reflection_steps):
            token_weights = reasoning.get("token_weights", None) if reasoning else None
            input_ids = reasoning.get("input_ids", None) if reasoning else None

            if token_weights is not None and input_ids is not None:
                weak_spans = self._extract_weak_spans(
                    input_ids, token_weights, self.tokenizer
                )
            else:
                weak_spans = []

            critique_prompt = (
                self._targeted_critique_prompt(weak_spans)
                if weak_spans else self._critique_prompt(current_answer)
            )

            critique, _ = self._generate(critique_prompt)
            revised, revised_reasoning = self._generate(
                self._revise_prompt(current_answer, critique)
            )

            conf_after = self._estimate_confidence(revised_reasoning)

            if conf_after > current_conf:
                current_answer = revised
                current_conf = conf_after
                result["reflected"] = True
                result["critique"] = critique

        # 4) Final metrics
        result["final_answer"] = current_answer
        result["confidence_after"] = current_conf
        result["confidence_delta"] = current_conf - conf_before
        result["reflection_helped"] = result["confidence_delta"] > 0.05

        return result
