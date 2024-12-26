import gradio as gr
import langchain_helper
import openai_helper


def get_chat_interface(enable_memory):
    return gr.ChatInterface(
        langchain_helper.get_chat_response_history,
        theme="soft",
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
        additional_inputs=[
            gr.Radio(
                [openai_helper.CHAT4O, openai_helper.GPT4O_MINI, openai_helper.O1, openai_helper.O1_MINI],
                value=openai_helper.CHAT4O,
                label="GPT Model",
            ),
            gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"),
            gr.Checkbox(value=enable_memory, label="Enable Memory"),
        ],
    )


def get_text_interface(system_prompt):
    return gr.ChatInterface(
        langchain_helper.get_text_process_response,
        theme="soft",
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
        additional_inputs=[
            system_prompt,
            gr.Radio(
                [openai_helper.GPT4O_MINI, openai_helper.GPT4O],
                value=openai_helper.GPT4O_MINI,
                label="GPT Model",
            ),
            gr.Slider(0, 2, value=openai_helper.DEFAULT_TEMP, label="Temperature"),
        ],
    )
