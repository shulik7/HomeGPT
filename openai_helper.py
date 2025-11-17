"""
OpenAI model configuration and constants.

This module defines available OpenAI models and their API identifiers.
"""

DEFAULT_TEMP = 0.7

# Current GPT-4 Models (November 2025)
GPT4O = "GPT-4o"
GPT4O_MINI = "GPT-4o mini"

# O1 Reasoning Models
O1 = "o1"
O1_MINI = "o1-mini"
O1_PREVIEW = "o1-preview"

# GPT-4 Turbo (Legacy but still useful)
GPT4_TURBO = "GPT-4 Turbo"

# Older GPT-4 (for specific needs)
GPT4 = "GPT-4"


MODEL_DICT = {
    GPT4O: "gpt-4o",
    GPT4O_MINI: "gpt-4o-mini",
    O1: "o1",
    O1_MINI: "o1-mini",
    O1_PREVIEW: "o1-preview",
    GPT4_TURBO: "gpt-4-turbo",
    GPT4: "gpt-4",
}

