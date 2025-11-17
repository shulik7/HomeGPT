"""LangChain helper functions for chat, translation, and document processing."""

import threading
from datetime import datetime, timedelta

import gradio as gr
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import (
    AsyncChromiumLoader,
    OnlinePDFLoader,
    YoutubeLoader,
)
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

import openai_helper

# Session-based memory management using simple message lists
_session_memories = {}
_memory_lock = threading.Lock()
_memory_last_access = {}

MEMORY_WINDOW_SIZE = 3


def get_session_memory(session_id):
    """Get or create message history for a specific session."""
    with _memory_lock:
        if session_id not in _session_memories:
            _session_memories[session_id] = []
        _memory_last_access[session_id] = datetime.now()
        return _session_memories[session_id]


def save_to_memory(session_id, human_message, ai_message):
    """Save a conversation turn to session memory with window limit."""
    with _memory_lock:
        if session_id not in _session_memories:
            _session_memories[session_id] = []
        
        _session_memories[session_id].append(HumanMessage(content=human_message))
        _session_memories[session_id].append(AIMessage(content=ai_message))
        
        # Keep only last N conversation turns (2 messages per turn)
        max_messages = MEMORY_WINDOW_SIZE * 2
        if len(_session_memories[session_id]) > max_messages:
            _session_memories[session_id] = _session_memories[session_id][-max_messages:]
        
        _memory_last_access[session_id] = datetime.now()


def cleanup_old_memories():
    """Remove memory instances that haven't been accessed in over 1 hour."""
    with _memory_lock:
        cutoff_time = datetime.now() - timedelta(hours=1)
        expired_sessions = [
            sid for sid, last_access in _memory_last_access.items()
            if last_access < cutoff_time
        ]
        for sid in expired_sessions:
            del _session_memories[sid]
            del _memory_last_access[sid]


def get_openai_model(model=openai_helper.GPT4O_MINI, temperature=openai_helper.DEFAULT_TEMP, api_key=None):
    """Create a ChatOpenAI instance with the specified model and settings."""
    # Map display name to API model identifier
    mapped_model = openai_helper.MODEL_DICT.get(model, model)
    return ChatOpenAI(
        model=mapped_model,
        temperature=temperature,
        api_key=SecretStr(api_key) if api_key else None
    )


def get_chat_response_history(
    message,
    history,
    model,
    temperature,
    enable_memory,
    api_key,
    request=None
):
    """Handle chat requests with conversation history support."""
    session_id = request.session_hash if request else "default"
    cleanup_old_memories()
    return get_response(message, "", model, temperature, enable_memory, api_key, session_id)


def get_text_process_response(
    message,
    history,
    system_prompt,
    model,
    temperature,
    api_key,
):
    """Handle text processing requests without memory."""
    return get_response(message, system_prompt, model, temperature, None, api_key, None)


def get_response(message, system_prompt, model, temperature, enable_memory, api_key, session_id):
    """Generate a response from the LLM with optional conversation memory using LCEL."""
    # O1 models require temperature of 1
    if model.startswith("o1"):
        temperature = 1
    
    llm = get_openai_model(model, temperature, api_key)
    
    # Build prompt messages
    messages = []
    
    # O1 models don't support system messages
    if not model.startswith("o1") and system_prompt:
        messages.append(SystemMessagePromptTemplate.from_template(system_prompt))
    
    # Add memory if enabled
    if enable_memory and session_id:
        session_history = get_session_memory(session_id)
        messages.append(MessagesPlaceholder(variable_name="history"))
        messages.append(HumanMessagePromptTemplate.from_template("{input}"))
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "input": message,
            "history": session_history
        })
        
        # Save to memory
        save_to_memory(session_id, message, response)
        return response
    else:
        messages.append(HumanMessagePromptTemplate.from_template("{input}"))
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"input": message})


def process_input(input_type, input_text, history, system_prompt, model, temperature, api_key):
    """Process different types of input (text, URL, PDF, YouTube)."""
    process_func = get_process_func(input_type)
    return process_func(input_text, history, system_prompt, model, temperature, api_key)


def get_process_func(input_type):
    """Get the appropriate processing function for the input type."""
    processors = {
        "Text": process_text,
        "Webpage URL": process_url,
        "PDF URL": process_online_pdf,
        "Youtube URL": process_youtube,
    }
    
    if input_type not in processors:
        raise ValueError(f"Unsupported input type: {input_type}")
        
    return processors[input_type]


def process_text(input_text, history, system_prompt, model, temperature, api_key):
    """Process plain text input."""
    response = get_response(input_text, system_prompt, model, temperature, None, api_key, None)
    history.append((input_text, response))
    return "", history


def translate(source_text, target_language, model, temperature, api_key):
    """Translate text to the target language."""
    system_prompt = get_translation_system_prompt(target_language)
    return get_response(source_text, system_prompt, model, temperature, None, api_key, None)


def get_translation_system_prompt(target_language):
    """Generate system prompt for translation tasks."""
    return (
        f"You are a Language Translator. Convert the user's input to {target_language}, "
        f"outputting only the translated text. "
        f"If the input is already in {target_language}, output it as-is."
    )


def process_url(url, history, system_prompt, model, temperature, api_key):
    """Process a webpage URL and extract content using LCEL."""
    try:
        llm = get_openai_model(model, temperature, api_key)
        prompt_template = system_prompt + "{text}"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        loader = AsyncChromiumLoader([url])
        html = loader.load()

        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            html, tags_to_extract=["p", "li", "div", "a"]
        )
        
        # Combine all document content
        combined_text = "\n\n".join([doc.page_content for doc in docs_transformed])
        response = chain.invoke({"text": combined_text})
        
        history.append((url, response))
        return "", history
    except Exception as e:
        error_msg = f"Error processing webpage: {str(e)}"
        history.append((url, error_msg))
        return "", history


def process_online_pdf(pdf_url, history, system_prompt, model, temperature, api_key):
    """Process an online PDF document using LCEL."""
    try:
        llm = get_openai_model(model, temperature, api_key)
        prompt_template = system_prompt + "{text}"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        loader = OnlinePDFLoader(pdf_url)
        pdf_data = loader.load()

        # Combine all document content
        combined_text = "\n\n".join([doc.page_content for doc in pdf_data])
        response = chain.invoke({"text": combined_text})
        
        history.append((pdf_url, response))
        return "", history
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        history.append((pdf_url, error_msg))
        return "", history



def process_youtube(youtube_url, history, system_prompt, model, temperature, api_key):
    """Process a YouTube video transcript using LangChain's YoutubeLoader."""
    try:
        llm = get_openai_model(model, temperature, api_key)
        prompt_template = system_prompt + "{text}"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        # Use LangChain's YoutubeLoader which properly handles youtube-transcript-api
        # Try English first, then auto-generated, then any available language
        loader = YoutubeLoader.from_youtube_url(
            youtube_url,
            add_video_info=False,
            language=["en", "en-US", "a.en"],
            translation="en"  # Translate to English if non-English transcript is used
        )
        docs = loader.load()
        
        if not docs:
            raise Exception("No transcript available for this video")
        
        # Combine document content
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        response = chain.invoke({"text": combined_text})
        
        history.append((youtube_url, response))
        return "", history
    except Exception as e:
        error_msg = f"Error processing YouTube video: {str(e)}\n\n"
        error_msg += "Possible reasons:\n"
        error_msg += "- Video has no captions/transcript available\n"
        error_msg += "- Video is private or region-restricted\n"
        error_msg += "- YouTube is blocking automated access\n"
        error_msg += "- Invalid video URL format\n\n"
        error_msg += f"URL attempted: {youtube_url}"
        history.append((youtube_url, error_msg))
        return "", history


