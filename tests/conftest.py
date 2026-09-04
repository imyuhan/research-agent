"""Shared test setup.

The application validates API settings during import. Tests use placeholder
values because no real provider calls should happen during unit tests.
"""

import os


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
