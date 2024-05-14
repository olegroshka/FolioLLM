import json
from datasets import Dataset

from src.dataset.etf_dataset import ETFDataset


def load_etf_dataset(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    dataset = ETFDataset(json_data)

    # Convert the ETFDataset to a Hugging Face Dataset
    hf_dataset = Dataset.from_dict({
        'text': [' '.join(str(value) for key, value in sample['features'].items()) for sample in dataset]
    })

    return hf_dataset