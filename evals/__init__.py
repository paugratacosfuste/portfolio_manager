"""Offline eval harness for the LLM patterns.

Each pattern in the app has a small set of golden-answer cases. The
runner feeds inputs through the live pipeline, then uses a Haiku judge
to score the output against a rubric. Results are written to
`evals/results/<timestamp>.json` for comparison across runs.

The purpose is not to replace unit tests — unit tests verify schema
conformance and mechanics, while evals verify that the pattern produces
*useful* output under realistic inputs. They're meant to be run
occasionally, not on every commit.
"""
