"""
OutputWriterAgent: Formats and writes the final prediction output file.
Output format per challenge spec: one CitizenID per line for citizens needing preventive support (label=1).
"""
from pathlib import Path
from typing import Dict


class OutputWriterAgent:
    """Writes the prediction output file."""

    def run(
        self,
        predictions: Dict[str, int],
        output_path: Path,
        verbose: bool = True,
    ) -> Path:
        at_risk = [cid for cid, label in predictions.items() if label == 1]
        at_risk.sort()  # Deterministic ordering

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for cid in at_risk:
                f.write(f"{cid}\n")

        if verbose:
            print(f"\n[OutputWriterAgent] Predictions written to: {output_path}")
            print(f"  Total citizens evaluated: {len(predictions)}")
            print(f"  Citizens flagged for preventive support (label=1): {len(at_risk)}")
            for cid, label in sorted(predictions.items()):
                status = "PREVENTIVE SUPPORT" if label == 1 else "standard monitoring"
                print(f"    {cid}: {status}")

        return output_path
