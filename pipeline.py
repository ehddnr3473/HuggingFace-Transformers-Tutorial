from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device_map="auto")

output = pipe("The secret to baking a good cake is", max_length=50, truncation=True)
print(output)