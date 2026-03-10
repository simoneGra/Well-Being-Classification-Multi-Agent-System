"""
WellbeingAnalysisAgent: LLM-based analysis of citizen wellbeing.

Token optimization strategy:
  - Citizens with rule_score >= CONFIDENT_RISK_THRESHOLD  → label=1 without LLM
  - Citizens with rule_score <= CONFIDENT_SAFE_THRESHOLD  → label=0 without LLM
  - Only ambiguous citizens (scores in between) → single batched LLM call
  - No training examples in LLM prompt (rule description in system prompt instead)
  - Minimal feature representation: only the 3 most discriminating signals

Result: 0 tokens when all citizens are clearly classified, otherwise minimal usage.
Tracked via Langfuse v3.
"""
import json
import re
from typing import Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.feature_engineer import CitizenFeatures
from config import (
    OPENROUTER_API_KEY, BASE_URL, MODEL_ID, TEMPERATURE, MAX_TOKENS,
    CONFIDENT_RISK_THRESHOLD, CONFIDENT_SAFE_THRESHOLD,
)


SYSTEM_PROMPT = (
    "Preventive health classifier. Rule: if a citizen has escalated care events "
    "(specialist/follow-up/emergency) AND declining physical activity AND rising "
    "environmental exposure → label 1. Otherwise → label 0. "
    "Respond ONLY with compact JSON: {\"ID\":0,...}. Zero extra text."
)


class WellbeingAnalysisAgent:
    """
    Classifies evaluation citizens with minimum token usage:
    1. Pre-classify all clearly confident cases with rule scores (0 tokens).
    2. Only call LLM for the ambiguous middle range (rare).
    3. LLM prompt contains only ambiguous citizens, no training examples.
    """

    def __init__(self, tracer=None):
        self._llm = None  # Lazy init — only create if LLM call is actually needed
        self.tracer = tracer

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=BASE_URL,
                model=MODEL_ID,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        return self._llm

    def run(
        self,
        train_features: Dict[str, CitizenFeatures],
        train_mobility,
        train_personas: str,
        eval_features: Dict[str, CitizenFeatures],
        eval_mobility,
        eval_personas: str,
        rule_predictions: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Returns {citizen_id: 0_or_1} for all evaluation citizens.
        Calls LLM only for citizens with ambiguous rule scores.
        """
        confident, ambiguous = self._split_by_confidence(eval_features)

        # Classify confident cases instantly (0 tokens)
        predictions = dict(confident)

        if not ambiguous:
            print(f"       [WellbeingAnalysisAgent] All {len(confident)} citizens classified by rules. LLM skipped (0 tokens).")
            return predictions

        # Only call LLM for ambiguous citizens
        print(f"       [WellbeingAnalysisAgent] {len(confident)} confident + {len(ambiguous)} ambiguous → calling LLM for {len(ambiguous)} citizens.")
        ambiguous_features = {cid: eval_features[cid] for cid in ambiguous}
        ambiguous_personas = _filter_personas(eval_personas, set(ambiguous))

        prompt = _build_minimal_prompt(ambiguous_features, ambiguous_personas)
        llm_results = self._call_llm(prompt, ambiguous_features)

        predictions.update(llm_results)
        return predictions

    def _split_by_confidence(
        self, eval_features: Dict[str, CitizenFeatures]
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Split citizens into:
          - confident: {cid: label} classified purely by rule score
          - ambiguous: [cid] that need LLM judgment
        """
        confident = {}
        ambiguous = []
        for cid, f in eval_features.items():
            if f.rule_risk_score >= CONFIDENT_RISK_THRESHOLD:
                confident[cid] = 1
            elif f.rule_risk_score <= CONFIDENT_SAFE_THRESHOLD:
                confident[cid] = 0
            else:
                ambiguous.append(cid)
        return confident, ambiguous

    def _call_llm(
        self, prompt: str, features: Dict[str, CitizenFeatures]
    ) -> Dict[str, int]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Track in Langfuse
        if self.tracer and raw:
            usage = response.usage_metadata or {}
            self.tracer.track_llm_call(
                name="wellbeing-llm-ambiguous",
                model=MODEL_ID,
                prompt=prompt,
                response=raw,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

        return _parse_response(raw, features)


# ---------------------------------------------------------------------------
# Prompt builder (minimal — only for ambiguous citizens)
# ---------------------------------------------------------------------------

def _build_minimal_prompt(
    features: Dict[str, CitizenFeatures],
    personas: str,
) -> str:
    """
    Ultra-compact prompt: only ambiguous citizens, only 3 key signals + persona bullets.
    Format: [ID] esc=N pai_slope=X eel=Y score=Z | health=... social=...
    """
    signals = _extract_risk_signals(personas)
    lines = []
    for cid, f in features.items():
        esc_types = ",".join(f.escalated_event_types) or "none"
        line = (
            f"[{cid}] esc={f.escalated_event_count}({esc_types}) "
            f"pai_slope={f.pai_slope:.1f} sqi_slope={f.sqi_slope:.1f} "
            f"eel={f.eel_mean:.0f}(slope={f.eel_slope:.1f}) "
            f"score={f.rule_risk_score:.0f}"
        )
        sig = signals.get(cid, "")
        if sig:
            line += f" | {sig}"
        lines.append(line)

    lines.append(f"\nClassify {list(features.keys())} → JSON only")
    return "\n".join(lines)


def _parse_response(raw: str, features: Dict[str, CitizenFeatures]) -> Dict[str, int]:
    """Parse LLM JSON response; fall back to rule predictions on failure."""
    if raw:
        try:
            match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                result = {}
                for cid in features:
                    val = parsed.get(cid)
                    result[cid] = int(val) if val is not None else features[cid].rule_prediction
                return result
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

    print("[WellbeingAnalysisAgent] WARNING: LLM parse failed, using rule fallback.")
    if raw:
        print(f"  Raw: {raw[:200]}")
    return {cid: f.rule_prediction for cid, f in features.items()}


# ---------------------------------------------------------------------------
# Persona helpers
# ---------------------------------------------------------------------------

def _extract_risk_signals(personas_text: str) -> Dict[str, str]:
    """Extract **Health behavior** and **Social pattern** bullet lines only."""
    signals = {}
    for part in personas_text.split("\n## ")[1:]:
        lines = part.strip().split("\n")
        if not lines:
            continue
        m = re.match(r'^([A-Z]{8})', lines[0])
        if not m:
            continue
        cid = m.group(1)
        parts = []
        for line in lines[1:]:
            if line.startswith("**Health behavior:**"):
                parts.append("health=" + line.split(":**", 1)[1].strip())
            elif line.startswith("**Social pattern:**"):
                parts.append("social=" + line.split(":**", 1)[1].strip())
        signals[cid] = " | ".join(parts)
    return signals


def _filter_personas(personas_text: str, citizen_ids: set) -> str:
    """Return persona text containing only the specified citizen IDs."""
    header = personas_text.split("\n## ")[0]
    sections = []
    for part in personas_text.split("\n## ")[1:]:
        m = re.match(r'^([A-Z]{8})', part.strip())
        if m and m.group(1) in citizen_ids:
            sections.append("## " + part)
    return header + "\n" + "\n".join(sections)
