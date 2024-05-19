import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, \
    T5ForConditionalGeneration
from src.dataset.data_utils import load_etf_dataset, load_prompt_response_dataset, load_etf_text_dataset
from src.eval.etf_advisor_evaluator import ETFAdvisorEvaluator
from src.training.etf_trainer import ETFTrainer, tokenize_structured_json, tokenize_prompt_response, tokenize_etf_text


class ETFAdvisorPipeline:
    def __init__(self, model_name, etf_structured_dataset, etf_prompt_response_dataset, test_prompts, output_dir, detailed=False):
        self.model_name = model_name
        self.etf_structured_dataset = etf_structured_dataset
        self.etf_prompt_response_dataset = etf_prompt_response_dataset
        self.test_prompts = test_prompts
        self.output_dir = output_dir
        self.detailed = detailed

    def run(self):
        # Step 1: Evaluate the base model
        #self.eval_base_model()

        # Step 2: Fine-tune the model
        self.finetune_model()

        # Step 3: Evaluate the fine-tuned model
        self.eval_finetuned_model()

    def eval_finetuned_model(self):
        print("\nEvaluating the fine-tuned model...")
        fine_tuned_model = self.load_finetuned_model()
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        fine_tuned_evaluator = ETFAdvisorEvaluator(fine_tuned_model, fine_tuned_tokenizer, self.test_prompts)
        fine_tuned_evaluator.evaluate(detailed=self.detailed)

    def finetune_model(self):
        # print("\nFine-tuning the model on prompt/response pairs...")
        # trainer_prompt_response = ETFTrainer(self.model_name, self.etf_prompt_response_dataset, tokenize_prompt_response)
        # trainer_prompt_response.tokenize_dataset()
        # trainer_prompt_response.train()
        # trainer_prompt_response.save_model(self.output_dir)

        print("\nFine-tuning the model on structured JSON...")
        trainer_structured_json = ETFTrainer(self.model_name, self.etf_structured_dataset, tokenize_etf_text)#tokenize_structured_json)
        # trainer_structured_json = ETFTrainer(self.output_dir, self.etf_structured_dataset, tokenize_structured_json)
        trainer_structured_json.tokenize_dataset()
        trainer_structured_json.train()
        trainer_structured_json.save_model(self.output_dir)

    def eval_base_model(self):
        print("Evaluating the base model...")
        base_model = self.load_base_model()
        base_tokenizer = self.create_tokenizer()
        base_evaluator = ETFAdvisorEvaluator(base_model, base_tokenizer, self.test_prompts)
        base_evaluator.evaluate(detailed=self.detailed)

    def create_tokenizer(self):
        if "t5" in self.model_name.lower():
            tokenizer = T5Tokenizer.from_pretrained(self.model_name).to('cuda')
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def load_base_model(self):
        if "t5" in self.model_name.lower():
            model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).to('cuda')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).to('cuda')

        return model

    def load_finetuned_model(self):
        if "t5" in self.model_name.lower():
            model = T5ForConditionalGeneration.from_pretrained(
                self.output_dir,
                self.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).to("cuda")

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.output_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).to('cuda')

        return model


def load_test_prompts(json_file):
    with open(json_file, 'r') as file:
        test_prompts = json.load(file)
    return test_prompts


def main():
    json_structured_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/etf_data_v2.json'
    json_prompt_response_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/etf_training_data.json'
    test_prompts_file = '/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/data/basic-competency-test-prompts-1.json'
    #model_name = 'bert-base-uncased'
    #model_name = 'FacebookAI/roberta-large'
    #model_name = "stabilityai/stablelm-2-zephyr-1_6b" #good(!)
    #model_name = "facebook/opt-125m"
    #model_name = "facebook/opt-350m"
    #model_name = "EleutherAI/gpt-neo-2.7B"
    #model_name = "gpt2"
    model_name = 'FINGU-AI/FinguAI-Chat-v1'
    #model_name = "gpt2-medium"
    #model_name = "gpt2-large"
    #model_name = 'google-t5/t5-large'
    #model_name = 'google/flan-t5-xxl'
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    output_dir = './fine_tuned_model/' + model_name
    detailed = True  # Set to False if you only want average scores

    #etf_structured_dataset = load_etf_dataset(json_structured_file)
    etf_structured_dataset = load_etf_text_dataset(json_structured_file)
    etf_prompt_response_dataset = load_prompt_response_dataset(json_prompt_response_file)
    test_prompts = load_test_prompts(test_prompts_file)

    pipeline = ETFAdvisorPipeline(model_name, etf_structured_dataset, etf_prompt_response_dataset, test_prompts, output_dir, detailed=detailed)
    pipeline.run()


if __name__ == '__main__':
    main()
