import json
import torch
import wandb
import logging

from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_utils import load_prompt_response_dataset, load_etf_text_dataset, load_prompt_response_dataset_adv
from src.eval.etf_advisor_evaluator import ETFAdvisorEvaluator
from src.eval.evaluator import ETFAdvisorEvaluatorGPT2, ETFAdvisorEvaluatorFingu
from src.training.etf_trainer import ETFTrainer, tokenize_etf_text, tokenize_prompt_response
from src.models.knowledge_aware_lora import KnowledgeAwareLoRAModel
from src.models.knowledge_aware_mora import KnowledgeAwareMoRAModel
from src.models.kolmogorov_arnold_lora import KolmogorovArnoldLoRAModel
from src.models.kolmogorov_arnold_mora import KolmogorovArnoldMoRAModel
from src.models.lora_model import LoRAModel
from src.models.mora_model import MoRAModel

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFAdvisorPipeline:
    def __init__(self, model_name, etf_structured_dataset, etf_prompt_response_dataset, test_prompts, output_dir,
                 detailed=False, mode="default", rank_config=None, knowledge_dim=128, hidden_features=128):
        self.model_name = model_name
        self.etf_structured_dataset = etf_structured_dataset
        self.etf_prompt_response_dataset = etf_prompt_response_dataset
        self.test_prompts = test_prompts
        self.output_dir = output_dir
        self.detailed = detailed
        self.mode = mode  # Mode can be "default", "lora", "mora", "knowledge_aware_lora", "knowledge_aware_mora", "kan_lora", "kan_mora"
        self.rank_config = rank_config
        self.knowledge_dim = knowledge_dim
        self.hidden_features = hidden_features

    def run(self,
            max_length=512,
            eval_steps=200,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            weight_decay=0.01,
            gradient_accumulation_steps=64):

        # Step 1: Load and evaluate the base model
        base_model, base_tokenizer = self.load_base_model()
        self.eval_model(base_model, base_tokenizer, "base")

        # Step 2: Fine-tune the model
        finetuned_model, finetuned_tokenizer = self.finetune_model(
            base_model,
            base_tokenizer,
            max_length,
            eval_steps,
            learning_rate,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            num_train_epochs,
            weight_decay,
            gradient_accumulation_steps
        )
        #finetuned_model, finetuned_tokenizer = self.load_finetuned_model()

        # Step 3: Evaluate the fine-tuned model
        self.eval_model(finetuned_model, finetuned_tokenizer, "finetuned")

    def eval_model(self, model, tokenizer, stage):
        print(f"\nEvaluating the {stage} model...")
        if 'gpt2' in model._get_name().lower():
            evaluator = ETFAdvisorEvaluatorGPT2(model, tokenizer, self.test_prompts, bert_score=True, rouge_score=False,
                                                perplexity=True, cosine_similarity=True)
        else:
            evaluator = ETFAdvisorEvaluatorFingu(model, tokenizer, self.test_prompts, bert_score=True, rouge_score=False,
                                                 perplexity=True, cosine_similarity=True)

        #evaluator = ETFAdvisorEvaluator(model, tokenizer, self.test_prompts, rouge_score=False)
        evaluator.evaluate(detailed=self.detailed)

    def finetune_model(self,
                       model,
                       tokenizer,
                       max_length=512,
                       eval_steps=200,
                       learning_rate=2e-5,
                       per_device_train_batch_size=1,
                       per_device_eval_batch_size=1,
                       num_train_epochs=3,
                       weight_decay=0.01,
                       gradient_accumulation_steps=64):
        print("\nFine-tuning the model on structured ETF data...")
        trainer_structured_json = ETFTrainer(model,
                                             tokenizer,
                                             self.etf_structured_dataset,
                                             tokenize_etf_text,
                                             self.test_prompts,
                                             max_length,
                                             eval_steps,
                                             learning_rate,
                                             per_device_train_batch_size,
                                             per_device_eval_batch_size,
                                             num_train_epochs,
                                             weight_decay,
                                             gradient_accumulation_steps)
        trainer_structured_json.tokenize_dataset()
        trainer_structured_json.train()
        trainer_structured_json.save_model(self.output_dir)

        finetuned_model, finetuned_tokenizer = self.load_finetuned_model()
        print("\nFine-tuning the model on prompt/response ETF data...")
        trainer_structured_json = ETFTrainer(finetuned_model,
                                             tokenizer,
                                             self.etf_prompt_response_dataset,
                                             tokenize_prompt_response,
                                             self.test_prompts,
                                             max_length,
                                             eval_steps,
                                             learning_rate,
                                             per_device_train_batch_size,
                                             per_device_eval_batch_size,
                                             num_train_epochs,
                                             weight_decay,
                                             gradient_accumulation_steps)
        trainer_structured_json.tokenize_dataset()
        trainer_structured_json.train()
        trainer_structured_json.save_model(self.output_dir)

        return finetuned_model, finetuned_tokenizer

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
                #attn_implementation="flash_attention_2",
                #torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # attn_implementation="flash_attention_2",
                # torch_dtype=torch.bfloat16
            )

        if self.mode == "lora":
            peft_config = LoraConfig(
                r=self.rank_config.get("r", 16),
                lora_alpha=self.rank_config.get("alpha", 64),
                lora_dropout=0.2,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
                ]
            )
            print("lora config:" + str(peft_config))
            model = get_peft_model(model, peft_config)
            #model = LoRAModel.from_pretrained(self.model_name)
        elif self.mode == "mora":
            model = MoRAModel.from_pretrained(self.model_name, rank_config=self.rank_config)
        elif self.mode == "knowledge_aware_lora":
            model = KnowledgeAwareLoRAModel.from_pretrained(self.model_name, knowledge_dim=self.knowledge_dim)
        elif self.mode == "knowledge_aware_mora":
            model = KnowledgeAwareMoRAModel.from_pretrained(self.model_name, rank_config=self.rank_config,
                                                            knowledge_dim=self.knowledge_dim)
        elif self.mode == "kan_lora":
            model = KolmogorovArnoldLoRAModel.from_pretrained(self.model_name, hidden_features=self.hidden_features)
        elif self.mode == "kan_mora":
            model = KolmogorovArnoldMoRAModel.from_pretrained(self.model_name, rank_config=self.rank_config,
                                                              hidden_features=self.hidden_features)

        model.to('cuda')
        tokenizer = self.create_tokenizer()
        return model, tokenizer

    def load_finetuned_model(self):
        if "t5" in self.model_name.lower():
            model = T5ForConditionalGeneration.from_pretrained(
                self.output_dir,
                attn_implementation="flash_attention_2",
                #torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.output_dir,
                attn_implementation="flash_attention_2",
                #torch_dtype=torch.bfloat16
            )

        if self.mode == "lora":
            peft_config = LoraConfig(
                r=self.rank_config.get("r", 16),
                lora_alpha=self.rank_config.get("alpha", 64),
                lora_dropout=0.2,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
                ]
            )
            model = get_peft_model(model, peft_config)
            #model = LoRAModel.from_pretrained(self.output_dir)
        elif self.mode == "mora":
            model = MoRAModel.from_pretrained(self.output_dir, rank_config=self.rank_config)
        elif self.mode == "knowledge_aware_lora":
            model = KnowledgeAwareLoRAModel.from_pretrained(self.output_dir, knowledge_dim=self.knowledge_dim)
        elif self.mode == "knowledge_aware_mora":
            model = KnowledgeAwareMoRAModel.from_pretrained(self.output_dir, rank_config=self.rank_config,
                                                            knowledge_dim=self.knowledge_dim)
        elif self.mode == "kan_lora":
            model = KolmogorovArnoldLoRAModel.from_pretrained(self.output_dir, hidden_features=self.hidden_features)
        elif self.mode == "kan_mora":
            model = KolmogorovArnoldMoRAModel.from_pretrained(self.output_dir, rank_config=self.rank_config,
                                                              hidden_features=self.hidden_features)

        model.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        return model, tokenizer

def load_test_prompts(json_file):
    with open(json_file, 'r') as file:
        test_prompts = json.load(file)
    return test_prompts

def run_pipeline(
        max_length=512,
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        gradient_accumulation_steps=64,
        model_name = 'FINGU-AI/FinguAI-Chat-v1',
        json_structured_file = '../../data/etf_data_v3_plain.json',
        test_prompts_file = '../../data/basic-competency-test-prompts-1.json',
        json_prompt_response_template_file = "../../data/training-template-adv.json",
        json_prompt_response_file_cleaned = "../../data/etf_data_v3_clean.json"):

    #model_name = 'gpt2'

    wandb.init(project="FolioLLM")  # Initialize wandb

    output_dir = './fine_tuned_model/' + model_name
    detailed = True  # Set to False if you only want average scores

    etf_structured_dataset = load_etf_text_dataset(json_structured_file)
    # etf_prompt_response_dataset = load_prompt_response_dataset(json_prompt_response_file)
    etf_prompt_response_dataset = load_prompt_response_dataset_adv(json_prompt_response_file_cleaned,
                                                                   json_prompt_response_template_file)
    test_prompts = load_test_prompts(test_prompts_file)

    rank_config = {
        "r": 128, #16
        "alpha": 512 #64
    }

    pipeline = ETFAdvisorPipeline(
        model_name,
        etf_structured_dataset,
        etf_prompt_response_dataset,
        test_prompts,
        output_dir,
        detailed=detailed,
        mode="lora",
        #mode="mora",
        #mode="kan_mora",
        rank_config=rank_config
    )

    pipeline.run()
    wandb.finish()

def main():
    run_pipeline()


if __name__ == '__main__':
    main()
