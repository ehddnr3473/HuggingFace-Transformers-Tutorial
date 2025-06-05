import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(device)

generated_ids = model.generate(**model_inputs, max_length=30)
output = tokenizer.batch_decode(generated_ids)[0]
print(output)
