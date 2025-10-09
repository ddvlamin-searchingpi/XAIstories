"""shapStories package

Narrative-driven XAI utilities to generate SHAP-based stories with LLMs.

Modules:
- `stories`: Core SHAPstory generation utilities.
- `llm_wrappers`: Lightweight wrappers for various LLM providers.
"""

__all__ = [
    "stories",
    "llm_wrappers",
]

from .stories import SHAPstory, unwrap


