import random
import json
import logging
from torch.utils.data import Dataset
import torch
from dataset_utils import get_value, translate_etf_entry

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

TEMPLATES_PATH = "../../data/training-template-adv.json"
ETFS_PATH = "../../data/etf_data_v3_clean.json"


class AdvETFDataset(Dataset):
    def __init__(self, etf_path, templates_path, sample_size):
        """
        Initialize the dataset by loading ETF data and templates.

        Args:
            etf_path (str): Path to the ETF data JSON file.
            templates_path (str): Path to the templates JSON file.
        """
        self.etf_data = self.load_json(etf_path)
        self.templates = self.load_json(templates_path)
        self.data = self.create_dataset(self.etf_data, self.templates, sample_size)

    def load_json(self, path):
        """
        Load JSON data from a file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Loaded JSON data.
        """
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON from {path}: {e}")
            return {}

    def create_dataset(self, etf_data, templates, sample_size):
        """
        Create a dataset by combining ETF data with templates.

        Args:
            etf_data (dict): Dictionary containing ETF data.
            templates (list): List of templates.
            sample_size (float): Fraction of the data to use for sampling.

        Returns:
            list: Combined dataset entries.
        """
        dataset = []

        random.shuffle(templates)
        sampled_etf_data = random.sample(etf_data, int(sample_size * len(etf_data)))

        for etf in sampled_etf_data:
            for template in templates:
                # etf = translate_etf_entry(etf)
                try:
                    prompt = template['prompt'].format(**etf)
                    response = template['response'].format(**etf)
                    dataset.append({'prompt': prompt, 'response': response})
                except KeyError as e:
                    logging.warning(f"Missing key in ETF data: {e}")

        print(f"Dataset size: {len(dataset)}")
        return dataset


    def get_all(self):
        return self.data

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Number of entries in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dataset entry at the specified index.
        """
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sample = self.data[idx]
        return sample

# Usage Example
if __name__ == "__main__":
    dataset = AdvETFDataset(ETFS_PATH, TEMPLATES_PATH, 1)
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        print(f"Sample entry: {dataset[i]}")
        print()

"""
Dataset size: 385600
Sample entry: {'prompt': 'Which fund has a 45.94, was incepted on 03/04/15, and focuses on European Region?', 'response': 'Consider the Wisdomtree Europe Hedged SmallCap Equity Fund. It has a NAV tracking error of 0.9399, a 30-day volume of 3076, and a return month-to-date of +4.06.'}

Sample entry: {'prompt': "What is the Wisdomtree Europe Hedged SmallCap Equity Fund's 0.58, 45.94, and 33145?", 'response': 'The Wisdomtree Europe Hedged SmallCap Equity Fund has an expense ratio of 0.58, class assets totaling 45.94, and a total value traded of 33145.'}

Sample entry: {'prompt': 'Could you provide the European Region, N.A., and N.A. for Wisdomtree Europe Hedged SmallCap Equity Fund?', 'response': 'The Wisdomtree Europe Hedged SmallCap Equity Fund focuses on the European Region region, specifically targeting the N.A. industry, and is associated with the N.A. economic sector.'}

Sample entry: {'prompt': 'For the Wisdomtree Europe Hedged SmallCap Equity Fund, what are its 243.0, 0.9399, and how has it performed YTD (+11.37)?', 'response': 'The Wisdomtree Europe Hedged SmallCap Equity Fund holds 243.0, has a NAV tracking error of 0.9399, and has achieved a YTD return of +11.37.'}

Sample entry: {'prompt': 'Can you find an ETF with a +0.66, focuses on Small-cap, and was incepted in 15?', 'response': 'The Wisdomtree Europe Hedged SmallCap Equity Fund is a match. It has a fund strategy of Blend, total fund assets of 45.94, and an expense ratio of 0.58.'}
"""

