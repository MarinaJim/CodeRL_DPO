from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-13b-Python-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Python-hf")
print("Resizing the model")
base_model.resize_token_embeddings(len(tokenizer))
adapter_dir = "exps/CodeLlama-13B-Python-hf-peft"

merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
print("Merging the adapter")
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained("exps/CodeLlama-13B-Python-hf-finetuned")
print("The model is saved!")