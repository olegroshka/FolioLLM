from bert_score import score
import numpy as np

class ETFAdvisorEvaluator:
    def __init__(self, model, tokenizer, test_prompts):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts

    def generate_response(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def evaluate(self, detailed=False):
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for prompt_data in self.test_prompts:
            prompt = prompt_data['prompt']
            expected_answer = prompt_data['expected_answer']

            generated_response = self.generate_response(prompt)

            # Calculate BERT score
            P, R, F1 = score([generated_response], [expected_answer], lang='en', verbose=False)
            precision_scores.append(P.mean().item())
            recall_scores.append(R.mean().item())
            f1_scores.append(F1.mean().item())

            if detailed:
                print(f"Prompt: {prompt}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Response: {generated_response}")
                print(f"BERT Score - Precision: {precision_scores[-1]:.4f}, Recall: {recall_scores[-1]:.4f}, F1: {f1_scores[-1]:.4f}")
                print("---")

        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        print(f"Average BERT Score - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
