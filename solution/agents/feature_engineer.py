"""
FeatureEngineerAgent: Computes quantitative risk features per citizen from raw event data.
No LLM calls - pure statistical computation in Python.

Key features:
- Metric means and standard deviations (high std = instability)
- Temporal trends (slope of linear regression over time)
- Presence of escalated event types (specialist/emergency = strong risk signal)
- Recent vs early trajectory comparison
- Composite risk score (rule-based)
"""
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from agents.data_loader import DataBundle, StatusEvent
from config import ESCALATED_EVENT_TYPES


@dataclass
class CitizenFeatures:
    citizen_id: str

    # Mean values
    pai_mean: float = 0.0
    sqi_mean: float = 0.0
    eel_mean: float = 0.0

    # Standard deviations (instability indicator)
    pai_std: float = 0.0
    sqi_std: float = 0.0
    eel_std: float = 0.0

    # Linear regression slopes (trend over time, per event index)
    pai_slope: float = 0.0
    sqi_slope: float = 0.0
    eel_slope: float = 0.0

    # Recent state (last 3 events)
    pai_recent: float = 0.0
    sqi_recent: float = 0.0
    eel_recent: float = 0.0

    # Early state (first 3 events)
    pai_early: float = 0.0
    sqi_early: float = 0.0
    eel_early: float = 0.0

    # Escalation signals
    has_escalated_events: bool = False
    escalated_event_count: int = 0
    escalated_event_types: List[str] = field(default_factory=list)

    # Event count
    total_events: int = 0

    # Composite rule-based risk score [0-100]
    rule_risk_score: float = 0.0

    # Rule-based pre-classification
    rule_prediction: int = 0


def _linear_slope(values: List[float]) -> float:
    """Compute slope of linear regression y ~ x where x = [0, 1, ..., n-1]."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


class FeatureEngineerAgent:
    """Computes per-citizen features from event data."""

    def run(self, bundle: DataBundle) -> Dict[str, CitizenFeatures]:
        citizen_events: Dict[str, List[StatusEvent]] = defaultdict(list)
        for ev in bundle.events:
            citizen_events[ev.citizen_id].append(ev)

        # Sort events by timestamp for each citizen
        for cid in citizen_events:
            citizen_events[cid].sort(key=lambda e: e.timestamp)

        features = {}
        for cid, events in citizen_events.items():
            features[cid] = self._compute_features(cid, events)

        return features

    def _compute_features(self, cid: str, events: List[StatusEvent]) -> CitizenFeatures:
        f = CitizenFeatures(citizen_id=cid)
        f.total_events = len(events)

        pais = [e.physical_activity_index for e in events]
        sqis = [e.sleep_quality_index for e in events]
        eels = [e.environmental_exposure_level for e in events]

        # Means
        f.pai_mean = statistics.mean(pais)
        f.sqi_mean = statistics.mean(sqis)
        f.eel_mean = statistics.mean(eels)

        # Standard deviations
        f.pai_std = statistics.stdev(pais) if len(pais) > 1 else 0.0
        f.sqi_std = statistics.stdev(sqis) if len(sqis) > 1 else 0.0
        f.eel_std = statistics.stdev(eels) if len(eels) > 1 else 0.0

        # Temporal slopes
        f.pai_slope = _linear_slope(pais)
        f.sqi_slope = _linear_slope(sqis)
        f.eel_slope = _linear_slope(eels)

        # Recent vs early
        n3 = min(3, len(events))
        f.pai_recent = statistics.mean(pais[-n3:])
        f.sqi_recent = statistics.mean(sqis[-n3:])
        f.eel_recent = statistics.mean(eels[-n3:])
        f.pai_early = statistics.mean(pais[:n3])
        f.sqi_early = statistics.mean(sqis[:n3])
        f.eel_early = statistics.mean(eels[:n3])

        # Escalated events
        escalated = [e.event_type for e in events if e.event_type in ESCALATED_EVENT_TYPES]
        f.has_escalated_events = len(escalated) > 0
        f.escalated_event_count = len(escalated)
        f.escalated_event_types = list(set(escalated))

        # Composite rule-based risk score
        f.rule_risk_score = self._compute_risk_score(f)
        f.rule_prediction = 1 if f.rule_risk_score >= 50.0 else 0

        return f

    def _compute_risk_score(self, f: CitizenFeatures) -> float:
        """
        Compute a rule-based risk score [0-100].
        Higher = more likely to need preventive support.
        """
        score = 0.0

        # Escalated event types are a very strong signal (max 40 points)
        if f.has_escalated_events:
            score += min(40.0, 15.0 + f.escalated_event_count * 8.0)

        # High variability in metrics (max 20 points)
        pai_var_score = min(10.0, f.pai_std * 0.8)
        sqi_var_score = min(10.0, f.sqi_std * 0.8)
        score += pai_var_score + sqi_var_score

        # Negative slopes (declining health metrics) (max 20 points)
        if f.pai_slope < 0:
            score += min(10.0, abs(f.pai_slope) * 1.5)
        if f.sqi_slope < 0:
            score += min(10.0, abs(f.sqi_slope) * 1.5)

        # High environmental exposure (max 10 points)
        if f.eel_mean > 55:
            score += min(10.0, (f.eel_mean - 55) * 0.5)

        # Rising environmental exposure (max 10 points)
        if f.eel_slope > 0:
            score += min(10.0, f.eel_slope * 1.5)

        # Low recent metrics (absolute floor) (max 10 points)
        if f.pai_recent < 35:
            score += min(5.0, (35 - f.pai_recent) * 0.3)
        if f.sqi_recent < 40:
            score += min(5.0, (40 - f.sqi_recent) * 0.3)

        return min(100.0, score)

    def summarize(self, features: Dict[str, CitizenFeatures]) -> str:
        """Return a compact text summary of all citizen features for LLM context."""
        lines = []
        for cid, f in features.items():
            lines.append(
                f"[{cid}] PAI: mean={f.pai_mean:.1f} std={f.pai_std:.1f} slope={f.pai_slope:.2f} | "
                f"SQI: mean={f.sqi_mean:.1f} std={f.sqi_std:.1f} slope={f.sqi_slope:.2f} | "
                f"EEL: mean={f.eel_mean:.1f} std={f.eel_std:.1f} slope={f.eel_slope:.2f} | "
                f"recent_PAI={f.pai_recent:.1f} recent_SQI={f.sqi_recent:.1f} recent_EEL={f.eel_recent:.1f} | "
                f"escalated_events={f.escalated_event_count}({','.join(f.escalated_event_types) or 'none'}) | "
                f"rule_score={f.rule_risk_score:.0f} rule_pred={f.rule_prediction}"
            )
        return "\n".join(lines)
