from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_int8_training

peft_path =


peft_model = AutoModelForCausalLM.from_pretrained("peft_model")
peft_model.save_pretrained("lora_adapter", save_adapter=True, save_config=True)
base_model = AutoModelForCausalLM.from_pretrained("base_model").to("cuda")
model_to_merge = PeftModel.from_pretrained(base_model, "lora_adapter")
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained("merged_model")
