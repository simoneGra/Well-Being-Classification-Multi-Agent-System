"""
WellbeingAnalysisAgent: LLM-based analysis of citizen wellbeing.

Performs a SINGLE batched LLM call covering all evaluation citizens at once.
The LLM receives:
  - Training citizens' condensed feature summaries + key persona signals
  - Evaluation citizens' condensed feature summaries + key persona signals
  - Instruction to classify each evaluation citizen as 0 or 1

Token efficiency: one call per run, batch all citizens.
Tracked via Langfuse v3.
"""
import json
import re
from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.feature_engineer import CitizenFeatures, FeatureEngineerAgent
from agents.mobility_analyzer import MobilityFeatures, MobilityAnalysisAgent
from config import OPENROUTER_API_KEY, BASE_URL, MODEL_ID, TEMPERATURE, MAX_TOKENS


SYSTEM_PROMPT = (
    "You are a preventive health AI. Classify citizens as 0 (standard monitoring) "
    "or 1 (needs preventive support). "
    "Respond ONLY with JSON: {\"CITIZENID\": 0_or_1, ...}. No other text."
)


class WellbeingAnalysisAgent:
    """
    Uses LLM to classify evaluation citizens using training data as calibration examples.
    Makes a single batched LLM call to minimize token usage.
    Tracked via Langfuse v3.
    """

    def __init__(self, tracer=None):
        self.llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=BASE_URL,
            model=MODEL_ID,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        self.tracer = tracer

    def run(
        self,
        train_features: Dict[str, CitizenFeatures],
        train_mobility: Dict[str, MobilityFeatures],
        train_personas: str,
        eval_features: Dict[str, CitizenFeatures],
        eval_mobility: Dict[str, MobilityFeatures],
        eval_personas: str,
        rule_predictions: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Returns {citizen_id: 0_or_1} for all evaluation citizens.
        One batched LLM call for maximum token efficiency.
        """
        prompt = self._build_compact_prompt(
            train_features, train_personas,
            eval_features, eval_personas,
            rule_predictions,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Track LLM call in Langfuse
        if self.tracer and raw:
            usage = response.usage_metadata or {}
            self.tracer.track_llm_call(
                name="wellbeing-classification-llm",
                model=MODEL_ID,
                prompt=prompt,
                response=raw,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

        return self._parse_response(raw, eval_features)

    def _build_compact_prompt(
        self,
        train_features: Dict[str, CitizenFeatures],
        train_personas: str,
        eval_features: Dict[str, CitizenFeatures],
        eval_personas: str,
        rule_predictions: Dict[str, int],
    ) -> str:
        """Build a compact prompt to minimize tokens while preserving key signals."""

        # Extract key persona signals (condensed to 3 lines max per citizen)
        train_signals = _extract_risk_signals(train_personas)
        eval_signals = _extract_risk_signals(eval_personas)

        lines = [
            "## TRAINING EXAMPLES (learn which pattern = label 1)",
            "",
            "Key: PAI=PhysicalActivity, SQI=SleepQuality, EEL=EnvExposure, slope=trend",
            "escalated_events = specialist/follow-up/emergency visits (STRONG risk signal)",
            "",
        ]

        for cid, f in train_features.items():
            signals = train_signals.get(cid, "")
            lines.append(
                f"[{cid}] escalated={f.escalated_event_count} types=[{','.join(f.escalated_event_types) or 'none'}] "
                f"PAI:{f.pai_mean:.0f}(slope={f.pai_slope:.1f}) SQI:{f.sqi_mean:.0f}(slope={f.sqi_slope:.1f}) "
                f"EEL:{f.eel_mean:.0f}(slope={f.eel_slope:.1f}) "
                f"recent_PAI={f.pai_recent:.0f} recent_EEL={f.eel_recent:.0f} "
                f"rule_score={f.rule_risk_score:.0f}"
            )
            if signals:
                lines.append(f"   persona: {signals}")

        lines += [
            "",
            "## EVALUATION CITIZENS (classify these as 0 or 1)",
            "",
        ]

        for cid, f in eval_features.items():
            signals = eval_signals.get(cid, "")
            lines.append(
                f"[{cid}] escalated={f.escalated_event_count} types=[{','.join(f.escalated_event_types) or 'none'}] "
                f"PAI:{f.pai_mean:.0f}(slope={f.pai_slope:.1f}) SQI:{f.sqi_mean:.0f}(slope={f.sqi_slope:.1f}) "
                f"EEL:{f.eel_mean:.0f}(slope={f.eel_slope:.1f}) "
                f"recent_PAI={f.pai_recent:.0f} recent_EEL={f.eel_recent:.0f} "
                f"rule_score={f.rule_risk_score:.0f}"
            )
            if signals:
                lines.append(f"   persona: {signals}")

        lines += [
            "",
            f"Rule-based scores: {rule_predictions}",
            "",
            f"Classify: {list(eval_features.keys())}",
            "Output JSON only: {\"CITIZENID\": 0, ...}",
        ]

        return "\n".join(lines)

    def _parse_response(self, raw: str, eval_features: Dict[str, CitizenFeatures]) -> Dict[str, int]:
        """Parse LLM JSON response with fallback to rule-based predictions."""
        if raw:
            try:
                match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    result = {}
                    for cid in eval_features:
                        val = parsed.get(cid)
                        if val is not None:
                            result[cid] = int(val)
                        else:
                            result[cid] = eval_features[cid].rule_prediction
                    return result
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass

        print("[WellbeingAnalysisAgent] WARNING: LLM response empty or unparseable, using rule-based fallback.")
        if raw:
            print(f"Raw response (first 200 chars): {raw[:200]}")
        return {cid: feat.rule_prediction for cid, feat in eval_features.items()}


def _extract_risk_signals(personas_text: str) -> Dict[str, str]:
    """
    Extract the **Mobility**, **Health behavior**, **Social pattern** summary lines
    from each persona section. These are the most token-efficient risk signal lines.
    """
    signals = {}
    parts = personas_text.split("\n## ")
    for part in parts[1:]:
        lines = part.strip().split("\n")
        if not lines:
            continue
        header = lines[0]
        cid_match = re.match(r'^([A-Z]{8})', header)
        if not cid_match:
            continue
        cid = cid_match.group(1)

        # Extract bold summary lines and first sentence of narrative
        key_parts = []
        for line in lines[1:]:
            if line.startswith("**Health behavior:**"):
                key_parts.append(line.replace("**Health behavior:** ", "health="))
            elif line.startswith("**Social pattern:**"):
                key_parts.append(line.replace("**Social pattern:** ", "social="))
            elif line.startswith("**Mobility:**"):
                key_parts.append(line.replace("**Mobility:** ", "mobility="))
        signals[cid] = " | ".join(key_parts)

    return signals
