import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
#model_id = "meta-llama/Meta-Llama-3-70B-Instruct"  # Adjust this line if necessary


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token="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"
)

pipeline("Hey how are you doing today?")
