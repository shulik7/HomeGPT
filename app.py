"""
YouTube Tools - AI-powered text processing application.

Features:
- Interactive chat with conversation memory
- Text translation to multiple languages
- Document processing (text, web pages, PDFs, YouTube videos)

All powered by OpenAI's GPT models with user-provided API keys.
"""

import gradio as gr

import gradio_helper
import langchain_helper
import openai_helper


# Example prompts for document processing
EXAMPLE_SYSTEM_PROMPTS = [
    "You are a Text Summarizer. Provide a concise summary of the input text within 100 words, capturing its essential points.",
    "You are a Grammar Checker. Correct grammatical errors in the input text while preserving its original meaning and style.",
    "You are a Content Optimizer. Your task is to polish the input text for better clarity, fluency, and impact, without altering its fundamental meaning.",
    "Please process the following webpage of a job posting and output the following information: company name, job title, salary range (use N/A if not mentioned), and ten skill names mentioned.",
    "Please list ten skill names mentioned in the following job description.",
    "Please provide the salary range mentioned in the following job description. If no salary range is specified, please inform the user accordingly.",
]

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🔒 Privacy & Security Notice
    - Your OpenAI API key is **never stored, logged, or saved** on our servers
    - API keys are only used temporarily for your current request and immediately discarded
    - Each user session is completely isolated - your conversations and API keys never mix with other users
    - All code is **open-source** and can be audited on HuggingFace/GitHub
    - We recommend reviewing the code if you have any concerns about data privacy
    """)
    
    with gr.Tab("Chat"):
        gradio_helper.get_chat_interface(True)

    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column():
                source_text = gr.Text(label="Input Text", lines=5)
            with gr.Column():
                target_text = gr.Text(label="Translated Text", lines=5)
                
        translate_btn = gr.Button("Translate")
        api_key_translate = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your OpenAI API key (sk-...)",
            type="password",
        )
        target_language = gr.Dropdown(
            ["English", "中文", "Español", "Français", "Deutsch"],
            label="Target Language",
            value="English",
        )
        model_selector = gr.Radio(
            [
                openai_helper.GPT4O_MINI,
                openai_helper.GPT4O,
                openai_helper.GPT4_TURBO,
            ],
            value=openai_helper.GPT4O_MINI,
            label="GPT Model",
        )
        temperature_slider = gr.Slider(
            0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"
        )
        translate_btn.click(
            langchain_helper.translate,
            inputs=[source_text, target_language, model_selector, temperature_slider, api_key_translate],
            outputs=[target_text],
        )

    with gr.Tab("Process Document"):
        system_prompt = gr.Textbox(
            value=EXAMPLE_SYSTEM_PROMPTS[0], label="System Prompt", lines=2
        )
        input_type = gr.Radio(
            ["Text", "Webpage URL", "PDF URL", "Youtube URL"],
            value="Text",
            label="Input Type",
        )
        chatbot = gr.Chatbot(label="History")
        input_text = gr.Text(label="Input", lines=2)

        submit_btn = gr.Button("Process")
        api_key_process = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your OpenAI API key (sk-...)",
            type="password",
        )
        model_selector = gr.Radio(
            [
                openai_helper.GPT4O_MINI,
                openai_helper.GPT4O,
                openai_helper.GPT4_TURBO,
            ],
            value=openai_helper.GPT4O_MINI,
            label="GPT Model",
        )
        temperature_slider = gr.Slider(
            0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"
        )
        gr.Examples(
            EXAMPLE_SYSTEM_PROMPTS,
            inputs=[system_prompt],
            label="Instruction Examples",
        )

        submit_btn.click(
            langchain_helper.process_input,
            inputs=[input_type, input_text, chatbot, system_prompt, model_selector, temperature_slider, api_key_process],
            outputs=[input_text, chatbot],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
