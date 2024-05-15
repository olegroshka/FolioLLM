import json
from datasets import Dataset
from src.dataset.etf_dataset import ETFDataset

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