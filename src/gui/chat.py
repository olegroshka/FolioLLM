import random
import re
import pickle
import sys
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.generation import TextStreamer
from sentence_transformers import SentenceTransformer
import os
from torch import nn
import faiss

from src.models.multitask import MultitaskLM
from src.optimization.optimization_mpt import optimizer

# from src.models.multitask import MultitaskLM
# from src.optimization.optimization_mpt import optimizer



# kostyli
# current_file_path = os.path.abspath(os.path.dirname(__file__))
current_file_path = os.path.abspath(os.getcwd())
optimization_path = os.path.abspath(os.path.join(current_file_path, '../optimization'))
models_path = os.path.abspath(os.path.join(current_file_path, '../models'))
sys.path.append(optimization_path)
sys.path.append(models_path)
with open('../../data/etf_data_short.pickle', 'rb') as file:
    etf_data = pickle.load(file)

INDEX_PATH = "../../data/etfs.index"
HEAD_PATH = '../pipeline/modules/class_head.pth'
SELECT_PATH = '../pipeline/modules/select_head.pth'
LORA_PATH = '../pipeline/fine_tuned_model/FINGU-AI/FinguAI-Chat-v1'
MODEL_NAME = "FINGU-AI/FinguAI-Chat-v1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

linear = nn.Linear(384, 2, bias=False)
linear.load_state_dict(torch.load(HEAD_PATH))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = MultitaskLM(MODEL_NAME, lora_path=LORA_PATH).to(device)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index(INDEX_PATH)

raw_context_message = (
    "You are a financial specialist specializing in ETF portfolio construction and optimization. "
    "Your role is to assist users by providing accurate, timely, and insightful information to guide their investment decisions. "
    "Consider their risk tolerance, investment goals, and market conditions when offering advice."
)

# Define generation parameters
generation_params = {
    'max_new_tokens': 200,
    'use_cache': True,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'eos_token_id': tokenizer.eos_token_id,
}

# Function to classify text
def optimization_prediction(text: str) -> int:
    logits = linear(torch.tensor(embedding_model.encode(text)))
    print(logits)
    return torch.argmax(logits).detach().item()


def extract_tickers(query):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, random.randint(3,8))
    print(distances, indices)
    return indices[0]


def optim_generation(user_input, history):
    print('optim body')
    indices = extract_tickers(user_input)
    etf_results = [etf_data[idx] for idx in indices]
    etf_context = "\n\n".join([
        f"{etf['ticker']} - {etf['etf_name']}\n{etf['description']}"
        for etf in etf_results
    ])

    initial_allocation = optimizer(  # should work by indices
        [etf['bbg_ticker'] for etf in etf_results], 
    )

    context = (
        "You are a financial specialist specializing in ETF portfolio construction and optimization. "
        "Your role is to assist users by providing accurate, timely, and insightful information to guide their investment decisions. "
        "Consider their risk tolerance, investment goals, and market conditions when offering advice."
        f"\n\nRelevant ETF Information:\n{etf_context}.\n"
    )

    print(initial_allocation)
    print(etf_context)
    print(indices)

    # history.append((user_input, f"some answer\n\n{initial_allocation}"))
    # return history


    # Construct the conversation messages
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_input},
    ]

    # Tokenize the chat messages
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    streamer = TextStreamer(tokenizer)

    # Generate the response
    outputs = model.model.generate(tokenized_chat, **generation_params, streamer=streamer)
    decoded_outputs = tokenizer.batch_decode(outputs)
    raw_answer = decoded_outputs[0]

    # Extract the assistant's response
    raw_answer = raw_answer.replace("<|im_end|>", "").replace("<|im_start|>", "")
    match = re.search(r"assistant\s*\n(.*?)(?=\nuser|\Z)", raw_answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = raw_answer.strip()

    # Append the new interaction to the history
    history.append((user_input, f"{answer}\n\n{initial_allocation}"))
    return history


def respond(inp, hist=[]):
    label = optimization_prediction(inp)
    if label == 0:
        print("executing optimization")
        return optim_generation(inp, hist)
    else:
        print("executing raw gen")
        return raw_generation(inp, hist)


def raw_generation(user_input, history):
    messages = [
        {"role": "system", "content": raw_context_message},
        {"role": "user", "content": user_input},
    ]
    # Tokenize the chat template
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    # Use a streamer for generating the response
    streamer = TextStreamer(tokenizer)

    # Generate the response
    outputs = model.model.generate(tokenized_chat, **generation_params, streamer=streamer)
    decoded_outputs = tokenizer.batch_decode(outputs)
    raw_answer = decoded_outputs[0]

    # Extract the assistant's response
    raw_answer = raw_answer.replace("<|im_end|>", "").replace("<|im_start|>", "")
    match = re.search(r"assistant\s*\n(.*?)(?=\nuser|\Z)", raw_answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = raw_answer.strip()

    # Append the new interaction to the history
    history.append((user_input, answer))
    return history


# Create the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="FolioLLM", height=1000)
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Type your message here...",
            #css="height: 100px;"
        )
        btn = gr.Button("Send")

    def submit_message(user_input, history=[]):
        new_history = respond(user_input, history)
        return new_history, ""


    btn.click(submit_message, [txt, chatbot], [chatbot, txt])

demo.launch()
