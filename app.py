import gradio as gr
import langchain_helper
import openai_helper
import gradio_helper


example_system_prompts = [
    "You are a Text Summarizer. Provide a concise summary of the input text within 100 words, capturing its essential points.",
    "You are a Grammar Checker. Correct grammatical errors in the input text while preserving its original meaning and style.",
    "You are a Content Optimizer. Your task is to polish the input text for better clarity, fluency, and impact, without altering its fundamental meaning.",
    "Please process the following webpage of a job posting and output the following information: company name, job title, salary range (use N/A if not mentioned), and ten skill names mentioned.",
    "Please list ten skill names mentioned in the follow job description",
    "Please provide the salary range mentioned in the following job description. If no salary range is specified, please inform the user accordingly.",
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
            [openai_helper.GPT3, openai_helper.GPT4],
            value=openai_helper.GPT3,
            label="GPT Model",
        )
        temperature = gr.Slider(
            0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"
        )
        submit.click(
            langchain_helper.translate,
            inputs=[source_text, target_language, model, temperature],
            outputs=[target_text],
        )

    with gr.Tab("Process Document"):
        system_prompt = gr.Textbox(
            value=example_system_prompts[0], label="System Prompt", lines=2
        )
        input_type = gr.Radio(
            ["Text", "Webpage URL", "PDF URL", "Youtube URL"],
            value="Text",
            label="Input Type",
        )
        chat_bot = gr.Chatbot(label="History")
        input = gr.Text(label="Input", lines=2)

        submit = gr.Button("Process")
        model = gr.Radio(
            [openai_helper.GPT3, openai_helper.GPT4],
            value=openai_helper.GPT3,
            label="GPT Model",
        )
        temperature = gr.Slider(
            0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"
        )
        gr.Examples(
            example_system_prompts,
            inputs=[system_prompt],
            label="Instruction Examples",
        )

        submit.click(
            langchain_helper.process_input,
            inputs=[input_type, input, chat_bot, system_prompt, model, temperature],
            outputs=[input, chat_bot],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
