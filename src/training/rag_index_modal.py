import re
import json
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from torch import nn

# from src.models.multitask import MultitaskLM


ETFS_PATH = "../../data/etf_data_v3_clean.json"
INDEX_PATH = "../../data/etfs.index"
MODEL_NAME = 'FINGU-AI/FinguAI-Chat-v1'
LORA_PATH = '../pipeline/fine_tuned_model/FINGU-AI/FinguAI-Chat-v1'
EMBEDDINGS_PATH = 'etf_embeddings.pth'
BATCH_SIZE = 30
GLOBAL_LIMIT = 100

def form(etf):
    top_sectors = {
        "Technology": etf.get("technology"),
        "Consumer Non-Cyclical": etf.get("consumer_non_cyclical"),
        "Communications": etf.get("communications"),
        "Financials": etf.get("financials")
    }

    top_sectors = {
        sector: float(value[:-1]) for sector, value in top_sectors.items()
        if value is not None
    }

    if not top_sectors:
        major_sector = "N/A"
        total_top_sectors = 0
    else:
        major_sector = max(top_sectors, key=top_sectors.get)
        total_top_sectors = sum(top_sectors.values())

    return f"""
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
- **Major Sector**: {major_sector} ({top_sectors.get(major_sector, 0)}%)
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


def main(etfs_path, index_path, model_name, lora_path, batch_size, global_limit, embeddings_path):
    with open(ETFS_PATH, 'r') as file:
        etf_data = json.load(file)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model.to(device)

    descriptions = [form(etf) for etf in etf_data][:GLOBAL_LIMIT]

    # Prepare descriptions
    num_batches = len(descriptions) // BATCH_SIZE + int(len(descriptions) % BATCH_SIZE != 0)
    
    # Initialize empty list to hold all embeddings
    all_embeddings = []
    for i in range(num_batches):
        print(i, '/', num_batches)
        batch_descriptions = descriptions[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        with torch.no_grad():
            embeddings = embedding_model.encode(batch_descriptions)
        # all_embeddings.append(embeddings.cpu().numpy())  # torch
        all_embeddings.append(embeddings)

    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    # Split code END

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    from faiss import write_index, read_index
    print(embeddings.shape)
    torch.save(embeddings, EMBEDDINGS_PATH)
    write_index(index, INDEX_PATH)


if __name__ == '__main__':
    main(
        etfs_path=ETFS_PATH,
        index_path=INDEX_PATH,
        model_name=MODEL_NAME,
        lora_path=LORA_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        batch_size=BATCH_SIZE,
        global_limit=GLOBAL_LIMIT)
