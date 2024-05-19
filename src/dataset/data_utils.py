import json
from datasets import Dataset
from src.dataset.etf_dataset import ETFDataset
from src.dataset.etf_text_dataset import ETFTextDataset


def load_test_prompts(json_file):
    with open(json_file, 'r') as file:
        test_prompts = json.load(file)
    return test_prompts

def load_etf_dataset(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    dataset = ETFDataset(json_data)

    # Convert the ETFDataset to a Hugging Face Dataset
    hf_dataset = Dataset.from_dict({
        'etf_ticker': [sample['etf_ticker'] for sample in dataset],
        'features': [sample['features'] for sample in dataset]
    })

    return hf_dataset

from datasets import Dataset

def load_etf_text_dataset(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    dataset = ETFTextDataset(json_data)

    # Convert the ETFTextDataset to a Hugging Face Dataset
    hf_dataset = Dataset.from_dict({
        'text': [sample['text'] for sample in dataset]
    })

    return hf_dataset

def load_prompt_response_dataset(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Convert the JSON data to a Hugging Face Dataset
    hf_dataset = Dataset.from_dict({
        'prompt': [sample['prompt'] for sample in json_data],
        'response': [sample['response'] for sample in json_data]
    })

    return hf_dataset
def main():
    json_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/etf_data.json'  # Replace with the path to your JSON file

    try:
        etf_dataset = load_etf_dataset(json_file)
        print("ETF dataset loaded successfully.")
        print(f"Number of samples: {len(etf_dataset)}")

        # Sanity check: Print the first sample
        print("\nFirst sample:")
        print(etf_dataset[0])

        # Sanity check: Print the keys of the features dictionary
        print("\nFeature keys:")
        print(list(etf_dataset[0]['features'].keys()))

    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{json_file}'.")
    except KeyError as e:
        print(f"Error: Missing key '{e.args[0]}' in the dataset.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()