
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

#input_text = "translate English to German: How old are you?"

input_texts = ["Suggest an ETF that focuses on technology stocks with low expense ratios.",
               "What are the best-performing ETFs in the healthcare sector over the past year?",
               "Recommend an ETF with a high dividend yield and low volatility."]

for input_text in input_texts:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print("Q:", input_text)
    print("A: ", tokenizer.decode(outputs[0]), "\n")

# out:
# Q: Suggest an ETF that focuses on technology stocks with low expense ratios.
# A:  <pad> iShares Russell 2000 Technology ETF (IWM)</s>
#
# Q: What are the best-performing ETFs in the healthcare sector over the past year?
# A:  <pad> XLV</s>
#
# Q: Recommend an ETF with a high dividend yield and low volatility.
# A:  <pad> iShares Russell 2000 Value ETF (IWM)</s>
