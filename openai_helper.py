import os

from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain, StuffDocumentsChain, summarize

from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import YoutubeLoader


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEFAULT_TEMP = 0.7
GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4-turbo-128k"

MODEL_DICT = {GPT3 : "gpt-3.5-turbo-1106", GPT4 : "gpt-4-1106-preview"}
