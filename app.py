import gradio as gr
from dotenv import load_dotenv, find_dotenv

from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# read env variables from the local .env file
_ = load_dotenv(find_dotenv())

DEFAULT_TEMP = 0.7
DEFAULT_MODEL = "gpt-3.5-turbo"

example_system_prompts = [
    "You are a Grammar Checker. Correct grammatical errors in the input text while preserving its original meaning and style.",
    "You are a Content Optimizer. Your task is to polish the input text for better clarity, fluency, and impact, without altering its fundamental meaning.",
    "You are a Text Summarizer. Provide a concise summary of the input text within 100 words, capturing its essential points.",
]


def get_openai_model(model=DEFAULT_MODEL, temperature=DEFAULT_TEMP):
    return ChatOpenAI(model=model, temperature=temperature)


def get_chat_response(
    message, history, system_prompt, model, temperature, enable_memory
):
    prompt = get_prompt(system_prompt, enable_memory)
    conversation = LLMChain(
        llm=get_openai_model(model, temperature),
        prompt=prompt,
        memory=memory if enable_memory else None,
    )
    return conversation.predict(input=message)


def get_prompt(system_prompt, enable_memory):
    messages = [SystemMessagePromptTemplate.from_template(system_prompt)]
    if enable_memory:
        messages.append(MessagesPlaceholder(variable_name="chat_history"))
    messages.append(HumanMessagePromptTemplate.from_template("{input}"))

    return ChatPromptTemplate(messages=messages)


def get_chat_interface(system_prompt, enable_memory):
    return gr.ChatInterface(
        get_chat_response,
        theme="soft",
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
        additional_inputs=[
            system_prompt,
            gr.Radio(
                ["gpt-3.5-turbo", "gpt-4"], value=DEFAULT_MODEL, label="GPT Model"
            ),
            gr.Slider(0, 2, value=DEFAULT_TEMP, label="Temperature"),
            gr.Checkbox(value=enable_memory, label="Enable Memory"),
        ],
    )


def translate(source_text, target_language, model, temperature):
    system_prompt = get_translation_system_prompt(target_language)
    return get_chat_response(source_text, _, system_prompt, model, temperature, False)


def get_translation_system_prompt(target_language):
    return (
        f"You are a Language Translator. Convert the user's input to {target_language}, outputting only the translated text."
        f"If the input is already in {target_language}, output it as-is."
    )


memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, k=8
)

with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        system_prompt = gr.Textbox(
            value="You are an AI Chat Assistant.",
            label="System Prompt",
            interactive=True,
        )

        get_chat_interface(system_prompt, True)
        gr.Examples(
            example_system_prompts,
            inputs=[system_prompt],
            label="System Prompt Examples",
        )

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
            ["gpt-3.5-turbo", "gpt-4"], value=DEFAULT_MODEL, label="GPT Model"
        )
        temperature = gr.Slider(0, 2, value=DEFAULT_TEMP, label="Temperature")
        submit.click(
            translate,
            inputs=[source_text, target_language, model, temperature],
            outputs=[target_text],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
