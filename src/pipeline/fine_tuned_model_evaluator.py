from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.dataset.data_utils import load_test_prompts, load_etf_dataset
from src.eval.t5_etf_advisor_evaluator import T5ETFAdvisorEvaluator


class T5FineTunedModelEvaluator:
    def __init__(self, model_name, test_prompts, output_dir, detailed=False):
        self.model_name = model_name
        self.test_prompts = test_prompts
        self.output_dir = output_dir
        self.detailed = detailed

    def run(self):
        # Step 1: Evaluate the base model
        # print("Evaluating the base model...")
        # base_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        # base_tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        # base_evaluator = T5ETFAdvisorEvaluator(base_model, base_tokenizer, self.test_prompts)
        # base_evaluator.evaluate(detailed=self.detailed)

        # Evaluate the fine-tuned model
        print("\nEvaluating the fine-tuned model...")
        fine_tuned_model = T5ForConditionalGeneration.from_pretrained(self.output_dir).to('cuda')
        fine_tuned_tokenizer = T5Tokenizer.from_pretrained(self.output_dir)
        fine_tuned_evaluator = T5ETFAdvisorEvaluator(fine_tuned_model, fine_tuned_tokenizer, self.test_prompts)
        fine_tuned_evaluator.evaluate(detailed=self.detailed)

def main():
    test_prompts_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/basic-competency-test-prompts.json'
    model_name = 'google-t5/t5-large'
    #model_name = 'google/flan-t5-xxl'
    output_dir = './fine_tuned_model'
    detailed = True  # Set to False if you only want average scores

    test_prompts = load_test_prompts(test_prompts_file)

    eval = T5FineTunedModelEvaluator(model_name, test_prompts, output_dir, detailed=detailed)
    eval.run()

if __name__ == '__main__':
    main()
