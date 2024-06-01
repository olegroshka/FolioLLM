import random
import re
import sys
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.generation import TextStreamer
import os

# kostyli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..', 'optimization')))
from optimization_mpt import optimizer


model_name = "FINGU-AI/FinguAI-Chat-v1"
# output_dir = '../pipeline/fine_tuned_model/' + model_name
output_dir = model_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(output_dir)
classifier = AutoModelForSequenceClassification.from_pretrained(output_dir, num_labels=2).to(device)

model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)

model.eval()
classifier.eval()


def optimization_prediction_ez(text: str) -> int:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    logits = classifier(**inputs).logits
    return torch.argmax(logits, dim=-1).item()

# Function to classify text
def optimization_prediction(text: str) -> int:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Perform forward pass to get logits
    with torch.no_grad():
        outputs = classifier(**inputs)
    
    # Get the logits and apply softmax to get probabilities; not needed
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted class (0 or 1)
    predicted_class = torch.argmax(probs, dim=-1).item()
       
    # Print the intermediate values for debugging
    print(f"Probabilities: {probs}")
    
    return predicted_class


def extract_tickers(history):
    #TODO
    return ['SPY US', 'IVY US', 'VO US', '510050 CH']


def optim_generation(user_input, history):
    tickers = extract_tickers(history)
    initial_allocation = optimizer(tickers, main=False) # main doesnt work

    context_message = "You are chatting with a helpful assistant."

    # Construct the conversation messages
    messages = [
        {"role": "system", "content": context_message},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": initial_allocation}
    ]

    # Prompt the model to explain the allocation
    follow_up_question = "Can you explain why these ETFs were chosen for the allocation?"
    messages.append({"role": "user", "content": follow_up_question})

    # Tokenize the chat messages
    tokenized_messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    # Define generation parameters
    generation_parameters = {
        'max_new_tokens': 100,
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
    outputs = model.generate(tokenized_messages, **generation_parameters, streamer=streamer)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_explanation = decoded_output[0]

    # Combine the initial allocation with the generated explanation
    combined_response = f"{initial_allocation}\n\n{generated_explanation.strip()}"

    # Append the conversation to history
    history.append((user_input, combined_response))


def respond(inp, hist=[]):
    label = optimization_prediction(inp)
    # print(f"classification label is {label}")

    if label == 0:
        # print("executing raw generation")
        return raw_generation(inp, hist)
    else:
        print("executing optimization")
        return optim_generation(inp, hist)


def raw_generation(user_input, history):
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
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(label="FolioLLM")
#     with gr.Row():
#         txt = gr.Textbox(show_label=False, placeholder="Type your message here...")  # Removed .style
#         btn = gr.Button("Send")


#     def submit_message(user_input, history=[]):
#         new_history = respond(user_input, history)
#         return new_history, ""


#     btn.click(submit_message, [txt, chatbot], [chatbot, txt])

#demo.launch()
