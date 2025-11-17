"""
Gradio interface components for the YouTube Tools application.

This module provides reusable Gradio interface components for:
- Chat interface with memory support
- Text processing interface
"""

import gradio as gr

import langchain_helper
import openai_helper


def get_chat_interface(enable_memory: bool) -> gr.ChatInterface:
    """
    Create a chat interface with optional conversation memory.
    
    Args:
        enable_memory: Whether to enable conversation history
        
    Returns:
        Configured Gradio ChatInterface
    """
    return gr.ChatInterface(
        langchain_helper.get_chat_response_history,
        type="messages",
        additional_inputs=[
            gr.Radio(
                [
                    openai_helper.GPT4O_MINI,
                    openai_helper.GPT4O,
                    openai_helper.O1_MINI,
                    openai_helper.O1,
                    openai_helper.O1_PREVIEW,
                    openai_helper.GPT4_TURBO,
                    openai_helper.GPT4,
                ],
                value=openai_helper.GPT4O_MINI,
                label="GPT Model",
            ),
            gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"),
            gr.Checkbox(value=enable_memory, label="Enable Memory"),
            gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API key (sk-...)",
                type="password",
            ),
        ],
    )


def get_text_interface(system_prompt: gr.Textbox) -> gr.ChatInterface:
    """
    Create a text processing interface for document analysis tasks.
    
    Args:
        system_prompt: Gradio Textbox component for system prompt input
        
    Returns:
        Configured Gradio ChatInterface
    """
    return gr.ChatInterface(
        langchain_helper.get_text_process_response,
        type="messages",
        additional_inputs=[
            system_prompt,
            gr.Radio(
                [
                    openai_helper.GPT4O_MINI,
                    openai_helper.GPT4O,
                    openai_helper.GPT4_TURBO,
                ],
                value=openai_helper.GPT4O_MINI,
                label="GPT Model",
            ),
            gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"),
            gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API key (sk-...)",
                type="password",
            ),
        ],
    )
