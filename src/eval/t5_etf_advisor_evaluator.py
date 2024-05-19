from bert_score import score
import numpy as np
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score

class T5ETFAdvisorEvaluator:
    def __init__(self, model, tokenizer, test_prompts):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts

    def generate_response(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.model.device)  # Move input tensors to the same device as the model
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

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
                'rouge-1': sum([score['rouge-1']['f'] for score in valid_rouge_scores]) / len(valid_rouge_scores),
                'rouge-2': sum([score['rouge-2']['f'] for score in valid_rouge_scores]) / len(valid_rouge_scores),
                'rouge-l': sum([score['rouge-l']['f'] for score in valid_rouge_scores]) / len(valid_rouge_scores)
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
