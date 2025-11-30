from os import getenv
from platform import processor
from typing import Dict, List, Optional

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import (
    Pipeline,
    pipeline,
)

load_dotenv(".env.local")

# Model constants
SMOL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LLAMA3_2_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Generation constants
MAX_NEW_TOKENS = 256
DEFAULT_ERROR_MESSAGE = "Whoops. Had some troubles. Mind trying again?"

def get_device() -> str:
    """
    Determine the best available device for model inference.
    
    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif processor() == "arm":
        return "mps"
    else:
        return "cpu"


# Initialize pipelines
device = get_device()

smol_pipe: Pipeline = pipeline("text-generation", model=SMOL_MODEL, device=device)

# If using a public gated model, you'll need to create a Hugging Face token with read access to
# that model and include it here. I included it in my environment variables
# (in a file that's been added to .gitignore) to keep it private
llama3_2_pipe: Pipeline = pipeline(
    "text-generation", model=LLAMA3_2_MODEL, device=device, token=getenv("HF_TOKEN")
)


def create_chat_message(role: str = "user", content: str = "") -> List[Dict[str, str]]:
    """
    Create a chat message in the format expected by the model.
    
    Args:
        role: The role of the message sender (default: "user")
        content: The content of the message
        
    Returns:
        List containing a single message dictionary
    """
    return [{"role": role, "content": content}]


def generate_response(
    pipe: Pipeline,
    message: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    truncation: bool = True
) -> str:
    """
    Generate a response using the specified pipeline.
    
    Args:
        pipe: The transformer pipeline to use
        message: The user's message
        max_new_tokens: Maximum number of tokens to generate
        truncation: Whether to truncate long inputs
        
    Returns:
        Generated text response or error message
    """
    try:
        message_as_chat = create_chat_message("user", message)
        
        response = pipe(
            message_as_chat,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
            truncation=truncation,
        )
        
        # Extract generated text from response
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and "generated_text" in response[0]:
                return response[0]["generated_text"]
        
        return DEFAULT_ERROR_MESSAGE
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return DEFAULT_ERROR_MESSAGE


def smol_predict(message: str, history: List[List[str]]) -> str:
    """
    Generate a response using the SMOL model.
    
    Args:
        message: The user's message
        history: Conversation history (unused but required by Gradio)
        
    Returns:
        Generated text response
    """
    return generate_response(smol_pipe, message, truncation=True)


def llama3_2_predict(message: str, history: List[List[str]]) -> str:
    """
    Generate a response using the Llama 3.2 model.
    
    Args:
        message: The user's message
        history: Conversation history (unused but required by Gradio)
        
    Returns:
        Generated text response
    """
    return generate_response(llama3_2_pipe, message, truncation=False)


def vote(like_data: gr.LikeData) -> None:
    """
    Handle user feedback on chatbot responses.
    
    Args:
        like_data: Gradio LikeData object containing vote information
    """
    if like_data.value and len(like_data.value) > 0:
        message = like_data.value[0]
        action = "liked" if like_data.liked else "disliked"
        print(f"Response {action}: {message}")


# Build the Gradio interface
with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# SMOL Chatbot")
            smol_chatbot = gr.Chatbot(
                placeholder="""# Hi! I'm Smolly 👋\n 
  ### 😊 A big brain in a little package. Ask Me Anything""",
                label="smol chatbot",
                resizable=True,
                avatar_images=(
                    None,
                    "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
                ),
            )
            smol_chatbot.like(vote, None, None)
            smol_chat = gr.ChatInterface(
                fn=smol_predict,
                chatbot=smol_chatbot,
                autofocus=True,
                examples=[
                    "What's the smallest model?",
                    "What's an LLM?",
                    "Where is Smallville?"
                ],
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("# llama3.2 Chatbot")
            llama_chatbot = gr.Chatbot(
                placeholder="""<h1>Me llamo <strong>Llama</strong> 🦙</h1><h3>😊 I like eating grass and answering questions. Ask Me Anything</h3>""",
                resizable=True,
                avatar_images=(
                    None,
                    "llama.jpg",
                ),
            )
            llama_chatbot.like(vote, None, None)
            llama_chat = gr.ChatInterface(
                fn=llama3_2_predict,
                chatbot=llama_chatbot,
                autofocus=False,
                examples=[
                    "Do llamas really like to eat grass?",
                    "Is llama fur soft?",
                    "What colors are llamas?",
                ],
            )


if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
