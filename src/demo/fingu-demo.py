#!pip install 'transformers>=4.39.0'
#!pip install -U flash-attn
#!pip install -q -U git+https://github.com/huggingface/accelerate.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,TextStreamer


model_id = 'FINGU-AI/FinguAI-Chat-v1'
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype= torch.bfloat16)
#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer)
model.to('cuda')


etf_data =  {
     "Ticker": "DIVP",
        "Name": "CULLEN ENHANCED EQT INCM ETF",
        "Manager": "Cullen Capital Management LLC",
        "Tot Ret Ytd": 572.631,
        "Tot Ret 1Y": 572.631,
        "Fund Asset Class Focus": "Equity",
        "ETFTickerName": "DIVP US Equity",
        "Summary": {
            "Name": "Cullen Enhanced Equity Income ETF",
            "Class Assets (MLN USD)": 7.76,
            "Fund Assets (MLN USD)": 7.76,
            "Expense Ratio": "0.55%",
            "YTD Return": "unavailable",
            "12M Yld": "+0.70%",
            "30D Vol": 4491,
            "YTD Flow": 6.34,
            "1M Flow": 1.28,
            "1 Yr NAV Trk Error": "unavailable",
            "Holdings": 47.0,
            "Primary": "Y",
            "Cross": "N"
        },
        "Flow": {
            "Currency, Security": "USD",
            "OAS Effective Duration": "unavailable",
            "OAS Duration Coverage Ratio": "unavailable",
            "YAS Modified Duration": "unavailable",
            "Options Available": "N",
            "Payment Type": "Income"
        },
        "Expense": {
            "Expense Ratio": "0.55%",
            "Fund Mgr Stated Fee": "0.55%",
            "Avg Bid Ask Spread": "0.47%",
            "1 Yr NAV Trk Error": "unavailable",
            "Premium": "+0.08%",
            "52W Avg Prem": "+0.05%"
        },
        "Regulatory": {
            "Fund Type": "ETF",
            "Structure": "Unknown",
            "Index Weight": "Unknown",
            "SFDR Class.": "unavailable",
            "Use Derivative": "Unknown",
            "Tax Form": "Unknown",
            "NAIC": "N",
            "UCITS": "N",
            "UK Reporting": "unavailable",
            "SFC": "N",
            "China": "N",
            "Leverage": "N",
            "Inception Date": "03/06/24"
        },
        "Performance": {
            "Name": "Cullen Enhanced Equity Income ETF",
            "1D Return": "+0.50%",
            "MTD Return": "+2.60%",
            "YTD Return": "unavailable",
            "1 Yr Return": "unavailable",
            "3 Yr Return": "unavailable",
            "5 Yr Return": "unavailable",
            "10 Yr Return": "unavailable",
            "12M Yld": "+0.70%"
        },
        "Liquidity": {
            "1D Vol": 3060,
            "Aggregated Volume": 3060,
            "Aggregated Value Traded": 79121,
            "Implied Liquidity": 30776785,
            "Bid Ask Spread": 0.13,
            "Short Interest%": "unavailable",
            "Open Interest": "unavailable"
        },
        "Industry": {
            "Materials": "4.63%",
            "Comm": "13.09%",
            "Cons Cyclical": "1.01%",
            "Cons Non-Cycl": "28.71%",
            "Divsf": "unavailable",
            "Energy": "11.38%",
            "Fin": "20.49%",
            "Ind": "7.65%",
            "Tech": "2.00%",
            "Utils": "8.25%",
            "Govt": "unavailable"
        },
        "Geography": {
            "N.Amer.": "85.61%",
            "LATAM": "unavailable",
            "West Euro": "11.59%",
            "APAC": "unavailable",
            "East Euro": "unavailable",
            "Africa/Middle East": "unavailable",
            "Central Asia": "unavailable"
        },
        "Descriptive": {
            "Economic Association": "unavailable",
            "Name": "CULLEN ENHANCED EQT INCM ETF",
            "Tot Ret Ytd": 572.631,
            "Fund Asset Class Focus": "Equity",
            "Fund Strategy": "Blend",
            "Fund Style": "unavailable",
            "General Attribute": "unavailable",
            "Local Objective": "unavailable",
            "Fund Objective": "Large-cap",
            "Fund Industry Focus": "unavailable",
            "Fund Geographical Focus": "United States"
        }
    }

messages = [
    {"role": "system","content": " you are as a finance specialist, help the user and provide accurat information."},
    #{"role": "user", "content": " what are the best approch to prevent loss?"},
    #{"role": "user", "content": "Explain what is ETF?"},
    {"role": "user", "content": "Can you describe DIVP ETF. Contex: Analyse detailed ETF provided below data and provide user in depth analisys: " + str(etf_data)},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

generation_params = {
    'max_new_tokens': 1000,
    'use_cache': True,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'eos_token_id': tokenizer.eos_token_id,
}

outputs = model.generate(tokenized_chat, **generation_params, streamer=streamer)
decoded_outputs = tokenizer.batch_decode(outputs)
decoded_outputs

'''
To avoid losses, it's essential to maintain discipline, set realistic goals, and adhere to predetermined rules for trading.
Diversification is key as it spreads investments across different sectors and asset classes to reduce overall risk.
Regularly reviewing and rebalancing positions can also ensure alignment with investment objectives. Additionally,
staying informed about market trends and economic indicators can provide opportunities for long-term capital preservation.
It's also important to stay patient and avoid emotional decision-making, as emotions often cloud judgment.
If you encounter significant losses, consider using stop-loss orders to limit your losses.
Staying disciplined and focusing on long-term objectives can help protect your investment portfolio from permanent damage.
'''
