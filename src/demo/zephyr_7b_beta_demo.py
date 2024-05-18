# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline


#stabilityai/stablelm-2-zephyr-1_6b
#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
pipe = pipeline("text-generation", model="stabilityai/stablelm-2-zephyr-1_6b", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages_list = [
    [
        {
            "role": "system",
            "content": "You are professional portfolio manager specializing in ETF who advises the client by provides deep inside into the finantial markets",
        },
        {"role": "user", "content": "Suggest an ETF that focuses on technology stocks with low expense ratios."},
    ],
    [
        {
            "role": "system",
            "content": "You are professional portfolio manager specializing in ETF who advises the client by provides deep inside into the finantial markets",
        },
        {"role": "user", "content": "What are the best-performing ETFs in the healthcare sector over the past year?"},
    ],
    [
        {
            "role": "system",
            "content": "You are professional portfolio manager specializing in ETF who advises the client by provides deep inside into the finantial markets",
        },
        {"role": "user", "content": "Recommend an ETF with a high dividend yield and low volatility."},
    ]
]

for messages in messages_list:
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(messages[1]["content"])
    print(outputs[0]["generated_text"], "\n")

# out:
#
# <|system|>
# You are professional portfolio manager specializing in ETF who advises the client by provides deep inside into the finantial markets</s>
# <|user|>
# Suggest an ETF that focuses on technology stocks with low expense ratios.</s>
# <|assistant|>
# One ETF that meets your criteria is the Invesco QQQ Trust (QQQ). It is a popular and widely recognized ETF that tracks the Nasdaq 100 Index,
# which is heavily weighted towards technology stocks. The expense ratio for QQQ is currently 0.20% per year, making it a relatively low-cost option
# for investors looking to gain exposure to technology stocks through an ETF. Some of the top holdings in QQQ include tech giants like Apple, Amazon,
# Microsoft, Google parent company Alphabet, and Facebook.
#
# <|system|>
# You are professional portfolio manager specializing in ETF who advises the client by provides deep inside into the finantial markets</s>
# <|user|>
# What are the best-performing ETFs in the healthcare sector over the past year?</s>
# <|assistant|>
# As of March 2021, the following are some of the best-performing ETFs in the healthcare sector over the past year:
#
# 1. Invesco Dynamic Pharmaceuticals ETF (PJP): This ETF focuses on pharmaceutical and biotechnology companies with a market capitalization of $1 billion or more.
# It has outperformed the S&P 500 Health Care Index by over 20% in the past year.
#
# 2. iShares U.S. Medical Devices ETF (IHI): This ETF invests in medical device and diagnostics companies. It has returned over 25% in the past year, significantly outperforming the S&P 500 Health Care Index.
#
# 3. Invesco NASDAQ Biotechnology ETF (NASB): This ETF focuses on biotechnology and pharmaceutical companies listed on the Nasdaq stock exchange. It has gained over 20% in the past year, outpacing the S&P 500 Health Care Index by over 15%.
#
# 4. Health Care Select Sector SPDR Fund (
#
# <|system|>
# You are professional portfolio manager specializing in ETF who advises the client by provides deep inside into the finantial markets</s>
# <|user|>
# Recommend an ETF with a high dividend yield and low volatility.</s>
# <|assistant|>
# One ETF that meets your criteria is the iShares Edge MSCI Min Vol USA ETF (USMV). It has a dividend yield of around 2.30% as of my last update,
# and its standard deviation (a measure of volatility) over the past year is 12.79%. This ETF invests in large and mid-cap US stocks with a focus on lower volatility,
# making it a good option for investors seeking a steady income stream with less risk. However, as with any investment, it('s important to conduct your own research '
# 'and consider your investment goals, risk tolerance, and time horizon before making any decisions.)


