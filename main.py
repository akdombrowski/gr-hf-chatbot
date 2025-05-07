import logging
from os import getenv
from platform import processor

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import (
    Pipeline,
    pipeline,
)

load_dotenv(".env.local")

CHAT_TEMPLATE = {"role": "user", "content": ""}
SMOL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LLAMA3_2_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# LLAMA3_2_MODEL = "google/flan-t5-base"

smol_pipe: Pipeline
llama3_2_pipe: Pipeline


if gr.NO_RELOAD:
    logger = logging.getLogger(f"{__name__}.predict")
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"{__name__}.predict---%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # "cuda" if torch detects availability, else "mps" if processor is "arm" (for Apple silicon)
    # otherwise, "cpu" is the default
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else "mps"
        if processor() == "arm"
        else "cpu"
    )
    logging.info("device=%s", device)

    smol_pipe = pipeline("text-generation", model=SMOL_MODEL, device=device)
    # If using a public gated model, you'll need to create a Hugging Face token with read access to
    # that model and include it here. I included it in my environtment variables
    # (in a file that's been added to .gitignore) to keep it private
    llama3_2_pipe = pipeline(
        "text-generation", model=LLAMA3_2_MODEL, device=device, token=getenv("HF_TOKEN")
    )


def log_info(message, *args):
    logger.info(message, *args)


def use_chat_template(role="user", message=""):
    template = CHAT_TEMPLATE
    CHAT_TEMPLATE["role"] = role
    CHAT_TEMPLATE["content"] = message
    return [template]


def smol_predict(message, history):
    message_as_chat = use_chat_template("user", message)

    # call pipeline
    response = smol_pipe(
        message_as_chat, return_full_text=False, max_new_tokens=256, truncation=True
    )

    # Default error message
    generated_text = "Whoops. Had some troubles. Mind trying again?"

    if len(response) > 0 and response[0]["generated_text"]:
        generated_text = "".join([m["generated_text"] for m in response])
    log_info("generated text response: \n%s\n", generated_text)

    return generated_text


def llama3_2_predict(message, history):
    message_as_chat = use_chat_template("user", message)

    response = llama3_2_pipe(
        message_as_chat,
        return_full_text=False,
        max_new_tokens=50,
        truncation=False,
    )

    log_info("generated text response: \n%s\n", response)
    # Default error message
    generated_text = "Whoops. Had some troubles. Mind trying again?"

    if response[0]["generated_text"]:
        generated_text = "".join([msg["generated_text"] for msg in response])

    return generated_text


def vote(like_data: gr.LikeData):
    message = like_data.value[0]
    was_liked = like_data.liked
    logger.info("\nbot msg:\n%s \nwas liked: \n%s \n", message, was_liked)
    if was_liked:
        print("Response liked: " + message)
    else:
        print("Response disliked: " + message)


with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# SMOL Chatbot")

    smol_chatbot = gr.Chatbot(
        type="messages",
        placeholder="""# Hi! I'm Smolly 👋\n 
### 😊 A big brain in a little package. Ask Me Anything""",
        # height="20vh",
        label="smol chatbot",
        min_height="10vh",
        max_height="40vh",
        resizable=True,
        avatar_images=(
            None,
            "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
        ),
        layout="panel",
        show_copy_all_button=True,
        watermark="built by frontegg",
    )
    smol_chatbot.like(vote, None, None)
    smol_chat = gr.ChatInterface(
        fn=smol_predict,
        type="messages",
        chatbot=smol_chatbot,
        autofocus=True,
        examples=["What's the smallest model?", "What's an LLM?", "Where is Smallville?"],
    )

    gr.Markdown("# llama3.2 Chatbot")

    llama_chatbot = gr.Chatbot(
        type="messages",
        placeholder="""<h1>Me llamo <strong>Llama</strong> 🦙</h1><h3>😊 I like eating grass and answering questions. Ask Me Anything</h3>""",
        resizable=True,
        min_height="10vh",
        max_height="40vh",
        avatar_images=(
            None,
            "llama.jpg",
        ),
        layout="panel",
        show_copy_all_button=True,
        watermark="built by frontegg",
    )
    llama_chatbot.like(vote, None, None)
    llama_chat = gr.ChatInterface(
        fn=llama3_2_predict,
        type="messages",
        chatbot=llama_chatbot,
        autofocus=False,
        examples=[
            "Do llamas really like to eat grasss?",
            "Is llama fur soft?",
            "What colors are llamas?",
        ],
    )


if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
