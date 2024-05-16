from bert_score import score
import numpy as np
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


class ETFAdvisorEvaluator:
    def __init__(self, model, tokenizer, test_prompts):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts

        # Set pad_token to eos_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        output = self.model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def evaluate(self, detailed=False):
        rouge = Rouge()
        vectorizer = TfidfVectorizer()

        bert_precision_scores = []
        bert_recall_scores = []
        bert_f1_scores = []
        rouge_scores = []
        cosine_similarities = []

        for prompt_data in self.test_prompts:
            prompt = prompt_data['prompt']
            expected_answer = prompt_data['expected_answer']

            generated_response = self.generate_response(prompt)

            # Calculate BERT scores
            bert_P, bert_R, bert_F1 = score([generated_response], [expected_answer], lang='en', verbose=False)
            bert_precision_scores.append(bert_P.mean().item())
            bert_recall_scores.append(bert_R.mean().item())
            bert_f1_scores.append(bert_F1.mean().item())

            # Calculate ROUGE scores
            rouge_score = rouge.get_scores(generated_response, expected_answer)
            if rouge_score:
                rouge_scores.append(rouge_score[0])
            else:
                rouge_scores.append(None)

            # Calculate cosine similarity
            tfidf_matrix = vectorizer.fit_transform([generated_response, expected_answer])
            cosine_sim = cosine_similarity(tfidf_matrix)[0][1]
            cosine_similarities.append(cosine_sim)

            if detailed:
                print(f"Prompt: {prompt}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Response: {generated_response}")
                print(
                    f"BERT Score - Precision: {bert_precision_scores[-1]:.4f}, Recall: {bert_recall_scores[-1]:.4f}, F1: {bert_f1_scores[-1]:.4f}")
                if rouge_score:
                    print(f"ROUGE Score: {rouge_score[0]}")
                else:
                    print("ROUGE Score: N/A")
                print(f"Cosine Similarity: {cosine_sim:.4f}")
                print("---")

        avg_bert_precision = sum(bert_precision_scores) / len(bert_precision_scores)
        avg_bert_recall = sum(bert_recall_scores) / len(bert_recall_scores)
        avg_bert_f1 = sum(bert_f1_scores) / len(bert_f1_scores)

        valid_rouge_scores = [score for score in rouge_scores if score is not None]
        if valid_rouge_scores:
            avg_rouge_score = {
                'rouge-1': np.mean([score['rouge-1']['f'] for score in valid_rouge_scores]),
                'rouge-2': np.mean([score['rouge-2']['f'] for score in valid_rouge_scores]),
                'rouge-l': np.mean([score['rouge-l']['f'] for score in valid_rouge_scores])
            }
        else:
            avg_rouge_score = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0
            }

        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)

        print(
            f"Average BERT Score - Precision: {avg_bert_precision:.4f}, Recall: {avg_bert_recall:.4f}, F1: {avg_bert_f1:.4f}")
        print(f"Average ROUGE Score: {avg_rouge_score}")
        print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}")

        return {
            "avg_bert_precision": avg_bert_precision,
            "avg_bert_recall": avg_bert_recall,
            "avg_bert_f1": avg_bert_f1,
            "avg_rouge_score": avg_rouge_score,
            "avg_cosine_similarity": avg_cosine_similarity
        }


# Example usage:
# Assuming you have a GPT-2 model and tokenizer loaded
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_prompts = [
    {"prompt": "What is the YTD return and expense ratio of Cullen Enhanced Equity Income ETF?",
     "expected_answer": "The Cullen Enhanced Equity Income ETF (DIVP US Equity) has a YTD return of 572.631% and an expense ratio of 0.55%."},
    # Add more test prompts here
]

evaluator = ETFAdvisorEvaluator(model, tokenizer, test_prompts)
evaluator.evaluate(detailed=True)
