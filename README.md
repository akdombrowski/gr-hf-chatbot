# gradio-huggingface-chatbot

How to build a chatbot with gradio, hugging face (and their transformers library).

## Create a Virtual Environment

**Highly recommended** 

Create a python virtual environment with `venv` or another virtual environment manager like [`uv`](https://docs.astral.sh/uv/pip/environments/#creating-a-virtual-environment) or [`virtualenv`](https://virtualenv.pypa.io/en/latest/user_guide.html#introduction)

### venv

```py
python3 -m venv {path/to/new/virtual/env}

# activate the environment (if not already active)
source {path/to/new/virtual}/bin/activate
```

### uv

```py
# creates environment in .venv
uv venv

# activate the environment
source .venv/bin/activate
```

### virtualenv

```py
# creates environment in ./env_name
virtualenv venv

# activate the environment
source .venv/bin/activate
```

## Install Dependencies

* **transformers**: Huggingface Transformers
* **gradio**: Gradio for app building
* **torch**: Pytorch
* (optional) **hf_xet**: Huggingface Xet for faster downloading

### pip or virtualenv

```sh
# optionally, add hf-xet as well
pip install -U transformers gradio torch
```

### uv

```sh
# optionally, add hf-xet as well
uv add transformers gradio torch
```

### Building with Gradio

Gradio provides a ChatInterface that makes it easy to create a quick chat interface. 
Hugging Face's Transformers library let's us use a variety of LLM's underneath.

### Hugging Face ChatInterface

```py
from gradio import gr

gr.ChatInterface()
```


Use chat template format for message `{"role": "user", "content": message}`.

### Chatbot models (LLMs) on Hugging Face

You can try some of the models out before choosing one with [Hugging Face's chat app](https://huggingface.co/chat/).

Search for LLMs using the following filters on [their models page](https://huggingface.co/models):

* task: **Text Generation** or **Text2Text Generation**
  * these are the tasks used for chatbot style AI Agents and only differ in the underlying techniques used for creating the LLM
* library: **Transformers**
  * important as we're using the transformers library

or use one of these links: 
* [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=trending)
* [Text2Text Generation](https://huggingface.co/models?pipeline_tag=text2text-generation&library=transformers&sort=trending)

Once you find one you want to try, open its page and copy the model id and name, or "path", at the top of the page:

e.g.,

* `HuggingFaceTB/SmolLM2-135M-Instruct` for https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
* `meta-llama/Llama-3.2-1B-Instruct` for https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Using High-Level Transformers Pipeline

The Pipelines in the Transformers library simplifies the process by handling a lot of the work behind the scenes, so you don't need to understand as much of the technical details of AI and LLMs. It's only required to give a model id and a *task*, although, even the task can sometimes be automatically inferred depending on the available info from the model. And, if you're using a model on Hugging Face, then you only need to give it the model path you got in the last step. Then using it requires calling the pipeline with the user's prompt or message. For a chat-style response, we'll want to use a chat template, which is simply a format that includes the role of the entity that the message belongs to.

For example,

```py
from transformers import pipeline

# use the pipeline utility method to initialize a text generation pipeline with the Llama 3.2 1B Instruct LLM
generate = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# ...
# once the user enters something into the chat, get the user's prompt, e.g., `user_prompt`
# ...

# call the pipeline with the user's prompt using the chat template
chat_message = [{"role": "user", "content": user_prompt}]
response = generate(chat_message, return_full_text=False)
```

`return_full_text` when set to `False` just tells the pipeline to only return the generated response. This is a convenience to make parsing it a little easier.


So, if the user enters "Is llama fur soft?", the `response` will look something like this:

```py
[{'generated_text': 'Llama fur is generally considered to be relatively soft and warm. Llamas are South American camelids, and their fur is known for its unique characteristics.'}]
```

which we can use the `generated_text` to respond in the chat as the agent.

Response from the text generation pipeline using a chat input format looks something like this:
```py
pipeline(user_chat_message, return_full_text=False)

# returns
[{'generated_text': 'I am a helpful AI assistant named SmolLM, trained by Hugging Face. I am here to assist you in various aspects of life, from personal to professional. Whether you are looking for advice on a specific topic or seeking help with a particular task, I am here to provide guidance and support.'}]
```

Tip: `max_new_tokens` parameter can be used to control the length of the response, but you might end up with sentences that are cut off. You can play around with the value or add something like a length penalty to make it more likely to end a response before reaching the max. The higher the max value, though, the longer it'll likely take to process.

## Run with Python or Gradio from Command Line

Run the app from the command line (making sure you’ve activated your virtual environment) with python {your_file}.py or if you want the app to reload when changes are detected in your file use the gradio command like this: gradio {your_file}.py! Then, open your browser and navigate to the url provided after running the command.

## Resources:

* [Gradio Docs: ChatInterface](https://www.gradio.app/docs/gradio/chatinterface)
* [Gradio Docs: Chatbot](https://www.gradio.app/docs/gradio/chatbot)
* [HF Docs: Transformers Pipelines](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/pipelines#pipelines)
* [HF Docs: Chat with Models](https://huggingface.co/docs/transformers/conversations)

Llama Image by wirestock on Freepik: https://www.freepik.com/free-photo/closeup-shot-white-llama-with-brown-face_17246819.htm#fromView=search&page=1&position=39&uuid=862928de-b947-4635-866d-7983846b9edc&query=llama