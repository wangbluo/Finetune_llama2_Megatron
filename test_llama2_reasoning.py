from transformers import pipeline

# Load the LLaMA model (assuming you have LLaMA 2 and its tokenizer installed)
model_name = "/home/wangbinluo/finetune_llama2_tp/output"  # Use the appropriate model name or path
generator = pipeline("text-generation", model=model_name)

# Define your prompt for the short story
prompt = """What are the three primary colors?"""

# Generate the short story
# Adjust max_length according to how long you want the story to be
story = generator(prompt, max_length=500, num_return_sequences=1)

# Print the generated story
print(story[0]["generated_text"])