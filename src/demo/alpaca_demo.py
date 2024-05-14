from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "chavinlo/alpaca-native"
#model_name = "allenai/open-instruct-stanford-alpaca-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


input_texts = ["Suggest an ETF that focuses on technology stocks with low expense ratios.",
               "What are the best-performing ETFs in the healthcare sector over the past year?",
               "Recommend an ETF with a high dividend yield and low volatility."]

for input_text in input_texts:
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), "\n")


# Suggest an ETF that focuses on technology stocks with low expense ratios.
# The ETF is the iShares S&P Global Technology ETF (IGV). It has an expense ratio of 0.49%. It has a portfolio of 105 technology stocks from around the world, including Apple, Microsoft, Amazon, and Alphabet.
# ETFs are a great way to invest in technology stocks because they provide
#
# What are the best-performing ETFs in the healthcare sector over the past year?
# The iShares U.S. Healthcare ETF (IYH) has gained 24.4% over the past year, while the SPDR Healthcare Select Sector ETF (XLV) has gained 22.4%. The iShares Nasdaq Biotechnology ETF (IBB) has gained 41.2%,
#
# Recommend an ETF with a high dividend yield and low volatility.
# The ETF should have a low expense ratio and a diversified portfolio of stocks. The ETF should also have a history of positive returns and low correlation to the overall market.
# ETFs to Consider The iShares Core S&P 500 ETF (IVV) is a low-cost, diversified ETF that tracks the S&P
