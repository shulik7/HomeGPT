import os
import gradio as gr
import langchain_helper
import openai_helper
import gradio_helper

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

example_system_prompts = [
    "You are a Text Summarizer. Provide a concise summary of the input text within 100 words, capturing its essential points.",
    "You are a Grammar Checker. Correct grammatical errors in the input text while preserving its original meaning and style.",
    "You are a Content Optimizer. Your task is to polish the input text for better clarity, fluency, and impact, without altering its fundamental meaning.",
]

example_webpage_process_instructions = [
    "Please process the following webpage of a job posting and output the following information: company name, job title, salary range (use N/A if not mentioned), and ten skill names mentioned.",
    "Please list ten skill names mentioned in the follow job description",
    "Please provide the salary range mentioned in the following job description. If no salary range is specified, please inform the user accordingly.",
]

example_youtube_summary_instructions = [
    "You are a Text Summarizer. Provide a concise summary of the input text within 300 words, capturing its essential points."
]



with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        gradio_helper.get_chat_interface(True)


    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column():
                source_text = gr.Text(label="Input Text", lines=5)
            with gr.Column():
                target_text = gr.Text(label="Translated Text", lines=5)
        submit = gr.Button("Translate")
        target_language = gr.Dropdown(
            ["English", "中文", "Español", "Français", "Deutsch"],
            label="Target Language",
            value="English",
        )
        model = gr.Radio(
            [openai_helper.GPT3, openai_helper.GPT4], value=openai_helper.GPT3, label="GPT Model"
        )
        temperature = gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature")
        submit.click(
            langchain_helper.translate,
            inputs=[source_text, target_language, model, temperature],
            outputs=[target_text],
        )

    with gr.Tab("Process Text"):
        system_prompt = gr.Textbox(

            lines=2,
            label="System Prompt",
            interactive=True,
        )

        model = gr.Radio(
            [openai_helper.GPT3, openai_helper.GPT4], value=openai_helper.GPT3, label="GPT Model"
        )

        temperature = gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature")
        chat_bot = gr.Chatbot()
        input_text = gr.Textbox(lines=2)
        submit = gr.Button("Submit")
        gr.Examples(
            example_system_prompts,
            inputs=[system_prompt],
            label="System Prompt Examples",
        )
        submit.click(
            langchain_helper.process_text,
            inputs=[input_text, chat_bot, system_prompt, model, temperature],
            outputs=[input_text, chat_bot],
        )

    with gr.Tab("Process Document"):
        system_prompt = gr.Textbox(
            value=example_system_prompts[0],
            label="System Prompt", 
            lines=2)
        input_type = gr.Radio(
            ["Text", "Webpage URL", "PDF URL", "Youtube URL"], value="Text", label="Input Type"
        )
        chat_bot = gr.Chatbot(label="History")
        input = gr.Text(label="Input", lines=2)

        submit = gr.Button("Process")
        model = gr.Radio(
            [openai_helper.GPT3, openai_helper.GPT4], value=openai_helper.GPT3, label="GPT Model"
        )
        temperature = gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature")
        gr.Examples(
            example_webpage_process_instructions,
            inputs=[system_prompt],
            label="Instruction Examples",
        )

        submit.click(
            langchain_helper.process_input,
            inputs=[input_type, input, chat_bot, system_prompt, model, temperature],
            outputs=[input_text, chat_bot],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
