import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, T5ForConditionalGeneration, \
    AutoModelForSeq2SeqLM, AutoModel, AutoModelForCausalLM

import torch

# Model options
models = {
    "BERT (Large)": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa": "deepset/roberta-base-squad2",
    "ALBERT": "twmkn9/albert-base-v2-squad2",
    "DistilBERT": "distilbert-base-uncased-distilled-squad",
    "XLNet": "xlnet-base-cased-squad2",
    "T5": "t5-small",
    "ELECTRA": "google/electra-small-generator-squad",
    #"Meta LLaMA 3-8B": "lama/meta-llama-3-8B",
    "Zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
    "gpt-j-6b": "EleutherAI/gpt-j-6b"
}

def load_model(model_name):
    if "t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif "zephyr" in model_name:  # Assuming Zephyr behaves like T5
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        #model = AutoModel.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        # Authentication for LLaMA or any model that requires it
        if "llama" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='your_api_key_here')
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token='your_api_key_here')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def answer_question(model_name, question, context):
    # Handle different models accordingly
    if "T5" in model_name:
        input_text = f"question: {question} context: {context}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=200, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif "Zephyr" in model_name:
        input_text = f"question: {question} context: {context}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=200, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer.strip()

def setup_ui(root):
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # Context input
    context_label = tk.Label(root, text="Context:")
    context_label.grid(row=0, column=0, sticky='ew')
    context_text = scrolledtext.ScrolledText(root, height=10)
    context_text.grid(row=1, column=0, sticky='nsew', rowspan=2)

    # Question input
    question_label = tk.Label(root, text="Ask a question:")
    question_label.grid(row=2, column=0, sticky='nsew')
    question_entry = scrolledtext.ScrolledText(root, height=5)
    question_entry.grid(row=3, column=0, sticky='nsew')

    # Model selection
    model_label = tk.Label(root, text="Select Model:")
    model_label.grid(row=4, column=0, sticky='nsew')
    model_var = tk.StringVar()
    model_selector = ttk.Combobox(root, textvariable=model_var, state='readonly')
    model_selector['values'] = list(models.keys())
    model_selector.grid(row=5, column=0, sticky='nsew')
    model_selector.set("BERT (Large)")

    model_selector.bind("<<ComboboxSelected>>", lambda e: on_model_change(model_selector.get()))

    # Answer display
    answer_label = tk.Label(root, text="Answer:")
    answer_label.grid(row=6, column=0, sticky='ew')
    answer_text = scrolledtext.ScrolledText(root, height=5, state='disabled')
    answer_text.grid(row=7, column=0, sticky='nsew')

    # Button to get answer
    answer_button = tk.Button(root, text="Get Answer", command=lambda: get_answer(question_entry, context_text, answer_text, model_selector.get()))
    answer_button.grid(row=8, column=0, sticky='ew')

def on_model_change(model_name):
    global tokenizer, model
    tokenizer, model = load_model(models[model_name])

def get_answer(question_entry, context_text, answer_text, model_name):
    question = question_entry.get("1.0", tk.END).strip()
    context = context_text.get("1.0", tk.END).strip()
    if question and context:
        answer = answer_question(model_name, question, context)
        answer_text.configure(state='normal')
        answer_text.delete("1.0", tk.END)
        answer_text.insert(tk.END, answer)
        answer_text.configure(state='disabled')

def main():
    root = tk.Tk()
    root.title("QA Chatbot")
    setup_ui(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# Contex samples
#
# Warren Buffett, known for his value investing strategy, focuses on companies with strong fundamentals, long-term growth potential, and stable earnings. In the Chinese market, similar investment opportunities might be found in companies like Alibaba Group Holding Limited, which is a major e-commerce and technology firm in China, and Tencent Holdings Limited, a leader in social media and gaming. Other potential investments could include JD.com, a major player in e-commerce and retail, and BYD Company Limited, which specializes in electric vehicles and batteries. These companies represent sectors such as technology, consumer goods, and green energy, aligning with Buffett's criteria of strong brand recognition, customer loyalty, and a durable competitive edge.
#
# Bill Ackman is known for his activist investing approach, focusing on companies that he believes are undervalued but have strong potential for improvement. In the Chinese market, this might include companies like Meituan, a leading platform for services such as food delivery and e-commerce, or Baidu, which has significant investments in artificial intelligence and internet services. NIO, an innovative electric vehicle manufacturer, could also be of interest due to its growth potential in the automotive sector. Ackman's strategy often involves significant stakes in a few companies rather than a diversified portfolio, emphasizing substantial improvements in company operations and governance.
#
#
# Q samples
#
# What are some Warren Buffett-style investment opportunities in the Chinese market?
# Which companies might Bill Ackman consider investing in within China?
# How do Alibaba's fundamentals align with value investing principles?