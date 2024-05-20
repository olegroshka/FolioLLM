import json
import logging
from torch.utils.data import Dataset
from etf_dataset import ETFDataset
from dataset_utils import snippet

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class ETFTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        text_snippet = self.create_text_snippet(sample)
        return {
            "text": text_snippet
        }

    def __len__(self):
        return len(self.data)

    def create_text_snippet(self, sample):
        return snippet.format(**sample)

    def get_value(self, sample, section, key):
        if section in sample and key in sample[section]:
            value = sample[section][key]
            if value == "N":
                return "No"
            elif value == "Y":
                return "Yes"
            else:
                return value
        else:
            return "N/A"

def main(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Create an instance of the ETFDataset
    dataset = ETFTextDataset(json_data)

    # Print the first 10 samples
    for i in range(10):
        sample = dataset[i]
        print(f"Sample {i+1}:")
        print(sample['text'])
        print()

if __name__ == '__main__':
    json_file = '../../data/etf_data_v3_plain.json'
    main(json_file)
