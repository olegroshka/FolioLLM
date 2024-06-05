import re
import json
import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the ETF data from JSON file
with open('../../data/etf_data_v2.json', 'r') as file:
    etf_data = json.load(file)

# Ensure all descriptions are strings and handle NaN values
# for etf in etf_data:
#     if not isinstance(etf.get("Description"), str):
#         etf["Description"] = ""
for etf in etf_data:
    for key, value in etf.items():
        if isinstance(value, float) and np.isnan(value):
            etf[key] = "Not Available"
        elif not isinstance(value, str):
            etf[key] = str(value)

# Load a pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the ETF descriptions
descriptions = [etf["Description"] for etf in etf_data]
embeddings = embedding_model.encode(descriptions)

# Convert embeddings to FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Store additional data for retrieval
etf_tickers = [etf["Ticker"] for etf in etf_data]

# Define the model name and the directory where the fine-tuned model is located
model_name = "FINGU-AI/FinguAI-Chat-v1"
output_dir = '../pipeline/fine_tuned_model/' + model_name

# Load the tokenizer and model from the fine-tuned directory
tokenizer = AutoTokenizer.from_pretrained(output_dir, attn_implementation="flash_attention_2")
base_model = AutoModelForCausalLM.from_pretrained(output_dir)
model = PeftModel.from_pretrained(base_model, output_dir)

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def search_etf(query, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [etf_data[idx] for idx in indices[0]]
    return results


def respond(user_input, history):
    # Search for relevant ETF information
    etf_results = search_etf(user_input)

    # etf_context = "\n".join([f"{etf['Ticker']}: {etf['Description']}" for etf in etf_results])

    # add all fields
    # etf_context = "\n\n".join([
    #     "\n".join([f"{key}: {etf[key]}" for key in etf.keys()])
    #     for etf in etf_results
    # ])

    etf_results = search_etf(user_input)
    etf_context = "\n\n".join([
        f"{etf['Ticker']} - {etf['Name']}\n{etf['Description']}"
        for etf in etf_results
    ])

    # Construct the context for the language model
    context = (
        "You are a financial specialist specializing in ETF portfolio construction and optimization. "
        "Your role is to assist users by providing accurate, timely, and insightful information to guide their investment decisions. "
        "Consider their risk tolerance, investment goals, and market conditions when offering advice."
        f"\n\nRelevant ETF Information:\n{etf_context}."
    )

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_input},
    ]

    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    # Define generation parameters
    generation_params = {
        'max_new_tokens': 1000,
        'use_cache': True,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'eos_token_id': tokenizer.eos_token_id,
    }

    # Use a streamer for generating the response
    streamer = TextStreamer(tokenizer)

    # Generate the response
    outputs = model.generate(tokenized_chat, **generation_params, streamer=streamer)
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
    chatbot = gr.Chatbot(label="FolioLLM (RAG)")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here...")  # Removed .style
        btn = gr.Button("Send")

    def submit_message(user_input, history):
        history = history or []
        new_history = respond(user_input, history)
        return new_history, ""

    btn.click(submit_message, [txt, chatbot], [chatbot, txt])

demo.launch()
