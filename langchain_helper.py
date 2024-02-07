import os
import openai_helper

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
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import YoutubeLoader

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, k=3
)

def try_map_model(model, model_dict):
    if model in model_dict:
        return model_dict[model]    
    return model


def get_openai_model(model=openai_helper.GPT3, temperature=openai_helper.DEFAULT_TEMP):
    return ChatOpenAI(model=try_map_model(model, openai_helper.MODEL_DICT), temperature=temperature)

def get_chat_response_history(
    message, history, model, temperature, enable_memory
):
    return get_response(message, "", model, temperature, enable_memory)

def get_text_process_response(
    message, history, system_prompt, model, temperature,
):
    return get_response(message, system_prompt, model, temperature, None)

def get_response(message, system_prompt, model, temperature, enable_memory):
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

def process_input(input_type, input, history, system_prompt, model, temperature):
    process_func = get_process_func(input_type)
    return process_func(input, history, system_prompt, model, temperature)


def get_process_func(input_type):
    if input_type == "Text":
        return process_text
    if input_type == "Webpage URL":
        return process_url
    if input_type == "PDF URL":
        return process_online_pdf
    if input_type == "Youtube URL":
        return process_youtube
    
    raise Exception(f"Unsupported input type:{input_type}")

def process_text(
    input, history, system_prompt, model, temperature,
):
    response = get_response(input, system_prompt, model, temperature, None)
    history.append((input, response))
    return "", history

def translate(source_text, target_language, model, temperature):
    system_prompt = get_translation_system_prompt(target_language)
    return get_response(source_text, system_prompt, model, temperature, None)


def get_translation_system_prompt(target_language):
    return (
        f"You are a Language Translator. Convert the user's input to {target_language}, outputting only the translated text."
        f"If the input is already in {target_language}, output it as-is."
    )

def process_url(input, history, system_prompt, model, temperature):
    prompt_template = system_prompt + "{text}"
    prompt = PromptTemplate.from_template(prompt_template)
    llm = get_openai_model(model, temperature)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, input_key="text")

    loader = AsyncChromiumLoader([input])
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        html, tags_to_extract=["p", "li", "div", "a"]
    )
    response = chain.run(docs_transformed)
    history.append((input, response))
    return "", history

def process_online_pdf(input, history, system_prompt, model, temperature):
    prompt_template = system_prompt + "{text}"
    prompt = PromptTemplate.from_template(prompt_template)
    llm = get_openai_model(model, temperature)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, input_key="text")

    loader = OnlinePDFLoader(input)
    pdf_dat = loader.load()

    response = chain.run(pdf_dat)
    history.append((input, response))
    return "", history

def process_youtube(input, history, system_prompt, model, temperature):
    prompt_template = system_prompt + "{text}"
    prompt = PromptTemplate.from_template(prompt_template)
    llm = get_openai_model(model, temperature)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, input_key="text")


    loader = YoutubeLoader.from_youtube_url(input, add_video_info=True)
    youtube_transcript = loader.load()

    response = chain.run(youtube_transcript)
    history.append((input, response))
    return "", history