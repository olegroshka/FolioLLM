import gradio as gr
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, RagRetriever, RagTokenizer, RagModel, RagConfig
import torch
import json
import os

# Define the model name and the directory where the fine-tuned model is located
model_name = "FinguAI-Chat-v1"
base_dir = '../pipeline/fine_tuned_model/FINGU-AI/'
output_dir = os.path.join(base_dir, model_name)
print(f"Output directory: {output_dir}")
print(os.listdir(output_dir))
# Ensure the path exists and contains necessary files
assert os.path.exists(output_dir), f"Directory does not exist: {output_dir}"

# Load the tokenizer from the fine-tuned directory
tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
print("Tokenizer loaded successfully")

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True)

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model.to(device)

# Load the ETF dataset
with open('../../data/etf_data_v2.json', 'r') as f:
    etf_data = json.load(f)

# Replace "Not Available" values with a valid value (e.g., 0)
for item in etf_data:
    for key, value in item.items():
        if value == "Not Available":
            item[key] = 0

etf_dataset = etf_data

# Define the configurations for the RAG model
rag_config = RagConfig(
    question_encoder=dict(
        model_type="qwen2",
        pretrained_model_name_or_path=output_dir,
    ),
    generator=dict(
        model_type="qwen2",
        pretrained_model_name_or_path=output_dir,
    ),
    model_type="rag-token",
    index_name="morningstar",  # Use Morningstar index
    index_path=None,
)
print("RAG Config created successfully")

print("Initializing RAG components...")
#rag_config = RagConfig.from_pretrained(output_dir)
rag_tokenizer = RagTokenizer(rag_config, tokenizer)
rag_retriever = RagRetriever(rag_config, tokenizer, tokenizer, index="morningstar")
rag_model = RagModel(rag_config, base_model).to(device)
print("RAG components initialized successfully")

# # Load tokenizers for question encoder and generator
# question_encoder_tokenizer = AutoTokenizer.from_pretrained(output_dir)
# generator_tokenizer = AutoTokenizer.from_pretrained(output_dir)
# print("Question Encoder and Generator Tokenizers loaded successfully")
#
# # Initialize the RAG components
# rag_retriever = RagRetriever.from_pretrained(
#     retriever_name_or_path=output_dir,
#     config=rag_config,
#     question_encoder_tokenizer=question_encoder_tokenizer,
#     generator_tokenizer=generator_tokenizer
# )
# print("RAG Retriever loaded successfully")
#
# rag_tokenizer = RagTokenizer.from_pretrained(output_dir)
# print("RAG Tokenizer loaded successfully")

rag_model = RagModel(rag_config, base_model).to(device)
print("RAG Model loaded successfully")

generation_params = {
    'max_new_tokens': 1000,
    'use_cache': True,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'eos_token_id': tokenizer.eos_token_id,
}


def respond(user_input, history):
    context = (
        "You are a financial specialist specializing in ETF portfolio construction and optimization. "
        "Your role is to assist users by providing accurate, timely, and insightful information to guide their investment decisions. "
        "Consider their risk tolerance, investment goals, and market conditions when offering advice."
    )
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_input},
    ]

    # Tokenize the chat template
    tokenized_chat = rag_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    # Retrieve relevant information from the ETF dataset
    retrieved_etf_data = rag_retriever(
        user_input,
        [item["features"] for item in etf_dataset],
        k=3  # Retrieve the top 3 most relevant ETF data
    )

    # Combine the chat template and the retrieved ETF data
    rag_input = {
        "input_ids": tokenized_chat.input_ids,
        "attention_mask": tokenized_chat.attention_mask,
        "context_input_ids": retrieved_etf_data.context_input_ids,
        "context_attention_mask": retrieved_etf_data.context_attention_mask,
    }

    # Generate the response using the RAG model
    outputs = rag_model.generate(**rag_input, **generation_params)
    decoded_outputs = rag_tokenizer.batch_decode(outputs)
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
    chatbot = gr.Chatbot(label="FolioLLM")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here...")
        btn = gr.Button("Send")

    def submit_message(user_input, history):
        history = history or []
        new_history = respond(user_input, history)
        return new_history, ""

    btn.click(submit_message, [txt, chatbot], [chatbot, txt])

demo.launch()