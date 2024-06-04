import re
import json
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from models.multitask import MultitaskLM
# exit(0)

ETFS_PATH = "../data/etf_data_v3_clean.json"
INDEX_PATH = "../data/etfs.index"
model_name = 'FINGU-AI/FinguAI-Chat-v1'

with open(ETFS_PATH, 'r') as file:
    etf_data = json.load(file)

embedding_model = MultitaskLM(model_name, index_path=INDEX_PATH)
tokenizer = AutoTokenizer.from_pretrained(model_name, )

def form(etf):
    top_sectors = {
        "Technology": etf["technology"],
        "Consumer Non-Cyclical": etf["consumer_non_cyclical"],
        "Communications": etf["communications"],
        "Financials": etf["financials"]
    }
    
    top_sectors = {
        sector: float(value[:-1]) for sector, value in top_sectors.items()
        if value is not None
    }
    
    major_sector = max(top_sectors, key=top_sectors.get)

    total_top_sectors = sum(top_sectors.values())

    return  f"""
The ETF's ticker is {etf['ticker']} ({etf['bbg_ticker']}), known as the {etf['etf_name']}.
{etf['description']}

**General Information:**
- **Fund Type**: {etf['fund_type']}
- **Manager**: {etf['manager']}
- **Asset Class Focus**: {etf['asset_class_focus']}
- **Fund Asset Group**: {etf['fund_asset_group']}
- **Geographical Focus**: {etf['fund_geographical_focus']}
- **Fund Objective**: {etf['fund_objective']}
- **Fund Strategy**: {etf['fund_strategy']}
- **Market Cap Focus**: {etf['fund_market_cap_focus']}

**Holdings and Allocations:**
- **Number of Holdings**: {etf['holdings']}
- **Major Sector**: {major_sector} ({top_sectors[major_sector]}%)
- **Top Sectors Total Allocation**: {total_top_sectors}%
- **Top Sectors**:
  - Technology: {etf['technology']}%
  - Consumer Non-Cyclical: {etf['consumer_non_cyclical']}%
  - Communications: {etf['communications']}%
  - Financials: {etf['financials']}%

**Geographic Allocation:**
- North America: {etf['north_america']}%
- Western Europe: {etf['western_europe']}%

**Additional Features:**
- **Options Available**: Yes
- **Payment Type**: {etf['payment_type']}
- **Structure**: {etf['structure']}
- **Inception Date**: {etf['inception_date']}"""

descriptions = [
    form(etf) for etf in etf_data
]

tokens = tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True)

embeddings = embedding_model.encode(**tokens)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
from faiss import write_index, read_index
write_index(index, INDEX_PATH)
