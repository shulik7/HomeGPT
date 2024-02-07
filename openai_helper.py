import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEFAULT_TEMP = 0.7
GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4-turbo-128k"

MODEL_DICT = {GPT3 : "gpt-3.5-turbo-1106", GPT4 : "gpt-4-1106-preview"}
