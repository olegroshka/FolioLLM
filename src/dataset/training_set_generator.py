import json
import argparse
import random
from dataset_utils import get_value, translate_etf_entry


def generate_training_data(etf_data, templates):
    training_data = []
    for etf in etf_data:
        # print(etf)
        for template in templates:
            try:
                prompt = template['prompt'].format(**etf)
                response = template['response'].format(**etf)
                # print(prompt)
                # print(response)
                training_data.append({"prompt": prompt, "response": response})
            except KeyError as e:
                print(f"Error formatting response for ETF {etf_name} ({ticker}): missing key {e}")

    return training_data

def main(etf_data_file, template_data_file, output_file, sample_size):
    # Load ETF data
    with open(etf_data_file, 'r') as f:
        etf_data = json.load(f)

    # Load templates
    with open(template_data_file, 'r') as f:
        templates = json.load(f)

    # Sample 10-20% of the ETF records
    sample_size = int(len(etf_data) * sample_size)
    sampled_records = random.sample(etf_data, sample_size) 

    # Generate training data
    training_data = generate_training_data(sampled_records, templates)

    # Save the training data
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=4)

    print(f"Prompt/response training data generated and saved to {output_file}. \nSample size: {sample_size}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for ETF advisor chatbot")
    parser.add_argument("sample_size", type=float, help="Sample size (e.g. 0.2)")
    parser.add_argument("etf_data_file", help="Path to the ETF data file")
    parser.add_argument("template_data_file", help="Path to the template data file")
    parser.add_argument("output_file", help="Path to the output file where training data will be saved")

    args = parser.parse_args()
    main(args.etf_data_file, args.template_data_file, args.output_file, args.sample_size)
