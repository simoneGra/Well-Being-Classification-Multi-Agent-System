"""
MirrorLife Well-being Classification System
Multi-Agent Pipeline for Preventive Health Support Prediction

Architecture:
  1. DataLoaderAgent        - Load all data sources (no LLM)
  2. FeatureEngineerAgent   - Compute quantitative risk features (no LLM)
  3. MobilityAnalysisAgent  - GPS-based mobility patterns (no LLM)
  4. WellbeingAnalysisAgent - LLM reasoning over features + personas (1 LLM call)
  5. OutputWriterAgent      - Write prediction file (no LLM)

Token optimization: single batched LLM call covering all citizens.
Monitored via Langfuse v3 Dashboard.
"""
import os
import sys
import time
import uuid
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Suppress non-critical Langfuse v3 OTEL context warnings before any imports
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)  # Temporarily suppress, re-enable below

sys.path.insert(0, str(Path(__file__).parent))

from config import TRAIN_DIR, EVAL_DIR, OUTPUT_DIR, TEAM_NAME, MODEL_ID
from agents.data_loader import DataLoaderAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.mobility_analyzer import MobilityAnalysisAgent
from agents.wellbeing_analyzer import WellbeingAnalysisAgent
from agents.output_writer import OutputWriterAgent
from utils.langfuse_tracker import get_langfuse_client, PipelineTracer


def get_trace_info(client, session_id):
    traces = []
    page = 1
    while True:
        response = client.api.trace.list(session_id=session_id, limit=100, page=page)
        if not response.data:
            break
        traces.extend(response.data)
        if len(response.data) < 100:
            break
        page += 1

    if not traces:
        return None

    observations = []
    for trace in traces:
        detail = client.api.trace.get(trace.id)
        if detail and hasattr(detail, 'observations'):
            observations.extend(detail.observations)

    if not observations:
        return None

    sorted_obs = sorted(
        observations,
        key=lambda o: o.start_time if hasattr(o, 'start_time') and o.start_time else datetime.min
    )

    counts = defaultdict(int)
    costs = defaultdict(float)
    total_time = 0

    for obs in observations:
        if hasattr(obs, 'type') and obs.type == 'GENERATION':
            model = getattr(obs, 'model', 'unknown') or 'unknown'
            counts[model] += 1
            if hasattr(obs, 'calculated_total_cost') and obs.calculated_total_cost:
                costs[model] += obs.calculated_total_cost
            if hasattr(obs, 'start_time') and hasattr(obs, 'end_time'):
                if obs.start_time and obs.end_time:
                    total_time += (obs.end_time - obs.start_time).total_seconds()

    first_input, last_output = "", ""
    if sorted_obs and hasattr(sorted_obs[0], 'input') and sorted_obs[0].input:
        first_input = str(sorted_obs[0].input)[:100]
    if sorted_obs and hasattr(sorted_obs[-1], 'output') and sorted_obs[-1].output:
        last_output = str(sorted_obs[-1].output)[:100]

    return {'counts': dict(counts), 'costs': dict(costs), 'time': total_time,
            'input': first_input, 'output': last_output}


def print_trace_info(info):
    if not info:
        print("\n  (No Langfuse traces found for this session)\n")
        return

    print("\n--- Langfuse Trace Summary ---")
    print("  LLM Calls by Model:")
    for model, count in info['counts'].items():
        print(f"    {model}: {count}")

    total_cost = sum(info['costs'].values())
    print("  Cost by Model:")
    for model, cost in info['costs'].items():
        print(f"    {model}: ${cost:.6f}")
    if total_cost > 0:
        print(f"    Total: ${total_cost:.6f}")

    print(f"  LLM Time: {info['time']:.2f}s")
    if info['input']:
        print(f"  Initial Input:  {info['input']}")
    if info['output']:
        print(f"  Final Output:   {info['output']}")


def run_pipeline(
    train_dir: Path = TRAIN_DIR,
    eval_dir: Path = EVAL_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    run_id = str(uuid.uuid4())[:8]
    session_id = f"{TEAM_NAME}-{eval_dir.name}-{run_id}"

    print("=" * 60)
    print("  MirrorLife Well-being Classification System")
    print("  Multi-Agent Preventive Health Pipeline")
    print("=" * 60)
    print(f"  Session : {session_id}")
    print(f"  Training: {train_dir}")
    print(f"  Eval    : {eval_dir}")
    print()

    # Re-enable logging for pipeline output
    logging.disable(logging.NOTSET)

    # Initialize Langfuse v3
    lf_client = get_langfuse_client()
    tracer = PipelineTracer(lf_client, session_id)

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # AGENT 1: Data Loader
    # ------------------------------------------------------------------ #
    print("[1/5] DataLoaderAgent: Loading training and evaluation data...")
    with tracer.agent_span("DataLoaderAgent"):
        loader = DataLoaderAgent()
        train_bundle = loader.run(train_dir)
        eval_bundle = loader.run(eval_dir)
        print(f"       Training  : {len(train_bundle.events)} events, {len(train_bundle.citizen_ids)} citizens: {train_bundle.citizen_ids}")
        print(f"       Evaluation: {len(eval_bundle.events)} events, {len(eval_bundle.citizen_ids)} citizens: {eval_bundle.citizen_ids}")

    # ------------------------------------------------------------------ #
    # AGENT 2: Feature Engineer
    # ------------------------------------------------------------------ #
    print("\n[2/5] FeatureEngineerAgent: Computing quantitative risk features...")
    with tracer.agent_span("FeatureEngineerAgent"):
        feat_eng = FeatureEngineerAgent()
        train_features = feat_eng.run(train_bundle)
        eval_features = feat_eng.run(eval_bundle)

        print("       Training features (risk indicators):")
        for cid, f in train_features.items():
            print(f"         {cid}: rule_score={f.rule_risk_score:.0f} pred={f.rule_prediction} "
                  f"escalated={f.escalated_event_count}({','.join(f.escalated_event_types) or 'none'}) "
                  f"pai_slope={f.pai_slope:.2f} sqi_slope={f.sqi_slope:.2f} eel_mean={f.eel_mean:.1f}")
        print("       Evaluation features (risk indicators):")
        for cid, f in eval_features.items():
            print(f"         {cid}: rule_score={f.rule_risk_score:.0f} pred={f.rule_prediction} "
                  f"escalated={f.escalated_event_count}({','.join(f.escalated_event_types) or 'none'}) "
                  f"pai_slope={f.pai_slope:.2f} sqi_slope={f.sqi_slope:.2f} eel_mean={f.eel_mean:.1f}")

    rule_predictions = {cid: f.rule_prediction for cid, f in eval_features.items()}

    # ------------------------------------------------------------------ #
    # AGENT 3: Mobility Analyzer
    # ------------------------------------------------------------------ #
    print("\n[3/5] MobilityAnalysisAgent: Analyzing GPS mobility patterns...")
    with tracer.agent_span("MobilityAnalysisAgent"):
        mob_agent = MobilityAnalysisAgent()
        train_mobility = mob_agent.run(train_bundle)
        eval_mobility = mob_agent.run(eval_bundle)
        for cid, mf in eval_mobility.items():
            print(f"         {cid}: {mf.mobility_summary}")

    # ------------------------------------------------------------------ #
    # AGENT 4: Wellbeing Analyzer (LLM - 1 batched call)
    # ------------------------------------------------------------------ #
    print("\n[4/5] WellbeingAnalysisAgent: LLM-based classification (1 batched call)...")
    with tracer.agent_span("WellbeingAnalysisAgent", input_data={"rule_predictions": rule_predictions}):
        analyzer = WellbeingAnalysisAgent(tracer=tracer)
        llm_predictions = analyzer.run(
            train_features=train_features,
            train_mobility=train_mobility,
            train_personas=train_bundle.personas,
            eval_features=eval_features,
            eval_mobility=eval_mobility,
            eval_personas=eval_bundle.personas,
            rule_predictions=rule_predictions,
        )
        print(f"       LLM predictions  : {llm_predictions}")

    # Consensus decision: combine rule-based and LLM predictions
    final_predictions = {}
    for cid in eval_features:
        rule_pred = rule_predictions.get(cid, 0)
        llm_pred = llm_predictions.get(cid, rule_pred)
        rule_score = eval_features[cid].rule_risk_score

        if rule_score >= 70:
            # Very high rule score → definite risk
            final_predictions[cid] = 1
        elif rule_score <= 10:
            # Very low rule score → definite safe
            final_predictions[cid] = 0
        elif rule_pred == llm_pred:
            # Both agree
            final_predictions[cid] = llm_pred
        else:
            # Disagreement: LLM gets the vote for nuanced contextual cases
            final_predictions[cid] = llm_pred

    print(f"       Final predictions: {final_predictions}")

    # ------------------------------------------------------------------ #
    # AGENT 5: Output Writer
    # ------------------------------------------------------------------ #
    print("\n[5/5] OutputWriterAgent: Writing prediction file...")
    output_path = output_dir / f"predictions_{eval_dir.name}.txt"
    with tracer.agent_span("OutputWriterAgent"):
        writer = OutputWriterAgent()
        writer.run(final_predictions, output_path)

    # Finalize Langfuse trace
    elapsed = time.time() - t0
    at_risk = [cid for cid, v in final_predictions.items() if v == 1]
    tracer.update_trace(
        output={"predictions": final_predictions, "at_risk_citizens": at_risk},
        metadata={"elapsed_seconds": round(elapsed, 2), "model": MODEL_ID},
    )
    tracer.flush()

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.2f}s")
    print(f"  Output: {output_path}")
    print(f"  Langfuse session: {session_id}")
    print(f"{'=' * 60}")

    try:
        trace_info = get_trace_info(lf_client, session_id)
        print_trace_info(trace_info)
    except Exception:
        pass

    return output_path


if __name__ == "__main__":
    output = run_pipeline()
    print(f"\n--- Final prediction file contents ---")
    with open(output) as f:
        content = f.read().strip()
    if content:
        print(content)
    else:
        print("(empty - no citizens flagged for preventive support)")
