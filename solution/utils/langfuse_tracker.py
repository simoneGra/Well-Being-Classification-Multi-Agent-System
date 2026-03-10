"""
Langfuse v3 tracker utility for monitoring token usage and agent traces.
Uses the Langfuse v3 SDK with OpenTelemetry-based tracing.
"""
from contextlib import contextmanager
from langfuse import Langfuse
from config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    TEAM_NAME,
)


def get_langfuse_client() -> Langfuse:
    """Initialize and return a configured Langfuse v3 client."""
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )


class PipelineTracer:
    """
    Wraps Langfuse v3 to provide per-agent span tracking for the pipeline.
    Uses start_as_current_span context manager for hierarchical traces.
    """

    def __init__(self, client: Langfuse, session_id: str):
        self.lf = client
        self.session_id = session_id

    @contextmanager
    def agent_span(self, name: str, input_data=None):
        """Context manager for tracking a single agent execution step."""
        with self.lf.start_as_current_span(
            name=name,
            input=input_data,
            metadata={"team": TEAM_NAME},
        ) as span:
            self.lf.update_current_trace(session_id=self.session_id, metadata={"team": TEAM_NAME})
            yield span

    def track_llm_call(
        self,
        name: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record a single LLM generation with token usage."""
        gen = self.lf.start_generation(
            name=name,
            model=model,
            input=prompt,
            output=response,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            metadata={"team": TEAM_NAME},
        )
        gen.end()
        return gen

    def update_trace(self, output=None, metadata=None):
        """Update the current trace with output and metadata (best-effort)."""
        try:
            # score_current_trace is the v3 way to attach metadata to trace
            if output:
                self.lf.score_current_trace(
                    name="pipeline-complete",
                    value=1.0,
                    comment=str(output),
                )
        except Exception:
            pass  # Non-critical

    def flush(self):
        """Flush all pending Langfuse events."""
        try:
            self.lf.flush()
        except Exception:
            pass
