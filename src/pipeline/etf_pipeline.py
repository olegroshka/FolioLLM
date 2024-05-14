import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.dataset.data_utils import load_etf_dataset
from src.eval.etf_advisor_evaluator import ETFAdvisorEvaluator
from src.training.etf_trainer import ETFTrainer


class ETFAdvisorPipeline:
    def __init__(self, model_name, etf_dataset, test_prompts, output_dir):
        self.model_name = model_name
        self.etf_dataset = etf_dataset
        self.test_prompts = test_prompts
        self.output_dir = output_dir

    def run(self):
        # Step 1: Evaluate the base model
        print("Evaluating the base model...")
        base_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        base_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_evaluator = ETFAdvisorEvaluator(base_model, base_tokenizer, self.test_prompts)
        base_evaluator.evaluate()

        # Step 2: Fine-tune the model
        print("\nFine-tuning the model...")
        trainer = ETFTrainer(self.model_name, self.etf_dataset)
        trainer.tokenize_dataset()
        trainer.train()
        trainer.save_model(self.output_dir)

        # Step 3: Evaluate the fine-tuned model
        print("\nEvaluating the fine-tuned model...")
        fine_tuned_model = AutoModelForMaskedLM.from_pretrained(self.output_dir)
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        fine_tuned_evaluator = ETFAdvisorEvaluator(fine_tuned_model, fine_tuned_tokenizer, self.test_prompts)
        fine_tuned_evaluator.evaluate()


def load_test_prompts(json_file):
    with open(json_file, 'r') as file:
        test_prompts = json.load(file)
    return test_prompts


def main():
    json_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/etf_data.json'
    test_prompts_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/basic-competency-test-prompts.json'
    model_name = 'bert-base-uncased'
    output_dir = './fine_tuned_model'

    etf_dataset = load_etf_dataset(json_file)
    test_prompts = load_test_prompts(test_prompts_file)

    pipeline = ETFAdvisorPipeline(model_name, etf_dataset, test_prompts, output_dir)
    pipeline.run()


if __name__ == '__main__':
    main()
