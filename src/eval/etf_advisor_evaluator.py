from bert_score import score
import numpy as np
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ETFAdvisorEvaluator:
    def __init__(self, model, tokenizer, test_prompts, bert_score=True, rouge_score=True, perplexity=True,
                 cosine_similarity=True):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.compute_bert_score = bert_score
        self.compute_rouge_score = rouge_score
        self.compute_perplexity = perplexity
        self.compute_cosine_similarity = cosine_similarity

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = 'left'

    def generate_response(self, prompt):
        messages = [
            {"role": "system",
             "content": "You are a professional portfolio manager specializing in ETF who advises the client by providing deep insight into the financial markets. Help the user and provide accurate information."},
            {"role": "user", "content": prompt},
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        generation_params = {
            'max_new_tokens': 1000,
            'use_cache': True,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        self.model.to("cuda")

        outputs = self.model.generate(tokenized_chat, **generation_params)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = decoded_outputs[0]#.split("assistant\n")[1].strip()

        return response

    def calculate_perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

    def evaluate(self, detailed=False):
        if self.compute_rouge_score:
            rouge = Rouge()
        if self.compute_cosine_similarity:
            vectorizer = TfidfVectorizer()

        bert_precision_scores = []
        bert_recall_scores = []
        bert_f1_scores = []
        rouge_scores = []
        cosine_similarities = []
        perplexity_scores = []

        for prompt_data in self.test_prompts:
            prompt = prompt_data['prompt']
            expected_answer = prompt_data.get('expected_answer', prompt_data.get('response'))

            generated_response = self.generate_response(prompt)

            if self.compute_bert_score:
                bert_P, bert_R, bert_F1 = score([generated_response], [expected_answer], lang='en', verbose=False)
                bert_precision_scores.append(bert_P.mean().item())
                bert_recall_scores.append(bert_R.mean().item())
                bert_f1_scores.append(bert_F1.mean().item())

            if self.compute_rouge_score:
                try:
                    rouge_score = rouge.get_scores(generated_response, expected_answer)
                    if rouge_score:
                        rouge_scores.append(rouge_score[0])
                    else:
                        rouge_scores.append(None)
                except ValueError as e:
                    print(
                        f"ROUGE score calculation failed for:\nGenerated Response: {generated_response}\nExpected Answer: {expected_answer}\nError: {e}")
                    rouge_scores.append(None)

            if self.compute_cosine_similarity:
                tfidf_matrix = vectorizer.fit_transform([generated_response, expected_answer])
                cosine_sim = cosine_similarity(tfidf_matrix)[0][1]
                cosine_similarities.append(cosine_sim)

            if self.compute_perplexity:
                perplexity = self.calculate_perplexity(generated_response)
                perplexity_scores.append(perplexity)

            if detailed:
                print(f"Prompt: {prompt}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Response: {generated_response}")
                if self.compute_bert_score:
                    print(
                        f"BERT Score - Precision: {bert_precision_scores[-1]:.4f}, Recall: {bert_recall_scores[-1]:.4f}, F1: {bert_f1_scores[-1]:.4f}")
                if self.compute_rouge_score:
                    print(f"ROUGE Score: {rouge_scores[-1] if rouge_scores[-1] else 'N/A'}")
                if self.compute_cosine_similarity:
                    print(f"Cosine Similarity: {cosine_sim:.4f}")
                if self.compute_perplexity:
                    print(f"Perplexity: {perplexity:.4f}")
                print("---")

        results = {}
        if self.compute_bert_score:
            results["avg_bert_precision"] = sum(bert_precision_scores) / len(bert_precision_scores)
            results["avg_bert_recall"] = sum(bert_recall_scores) / len(bert_recall_scores)
            results["avg_bert_f1"] = sum(bert_f1_scores) / len(bert_f1_scores)

        if self.compute_rouge_score:
            valid_rouge_scores = [score for score in rouge_scores if score is not None]
            if valid_rouge_scores:
                results["avg_rouge_score"] = {
                    'rouge-1': np.mean([score['rouge-1']['f'] for score in valid_rouge_scores]),
                    'rouge-2': np.mean([score['rouge-2']['f'] for score in valid_rouge_scores]),
                    'rouge-l': np.mean([score['rouge-l']['f'] for score in valid_rouge_scores])
                }
            else:
                results["avg_rouge_score"] = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

        if self.compute_cosine_similarity:
            results["avg_cosine_similarity"] = sum(cosine_similarities) / len(cosine_similarities)

        if self.compute_perplexity:
            results["avg_perplexity"] = sum(perplexity_scores) / len(perplexity_scores)

        print(results)
        return results

# Example usage:
# Assuming you have a GPT-2 model and tokenizer loaded
# model_name = "gpt2"
# model_name = "FINGU-AI/FinguAI-Chat-v1"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# test_prompts = [
#     {"prompt": "What is the YTD return and expense ratio of Cullen Enhanced Equity Income ETF?",
#      "expected_answer": "The Cullen Enhanced Equity Income ETF (DIVP US Equity) has a YTD return of 572.631% and an expense ratio of 0.55%."},
#     # Add more test prompts here
# ]
#
# evaluator = ETFAdvisorEvaluator(model, tokenizer, test_prompts, bert_score=True, rouge_score=False, perplexity=True, cosine_similarity=True)
# evaluator.evaluate(detailed=True)
