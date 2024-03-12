from transformers import LlamaForCausalLM, AutoTokenizer

# 1024.json
instruction = """Have a function that identifies the locations of all consonants in the provided string. Then, modify the code to handle various edge cases such as multiple spaces between words, punctuation marks, and numbers. \n\nThe provided sentence is: synchronization.\n\nAlso debug an erroneous code that seems to be attempting to solve the problem but contains mistakes. Here is the code snippet:\n\n```python\ndef consonant_locations(sentence):\n    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']\n    locations = []\n    for i in range(sentence):\n        if sentence[i] in consonants:\n            locations.append(i)\n    return locations\nprint(consonant_locations('synchronization'))\n```\n\nCorrect the code and explain the errors made in the initial code snippet."""
output = """The erroneous code seems to be trying to solve the problem correctly but made several mistakes. First, it fails to correctly iterate over the characters in the sentence because in the `for` loop, `range(sentence)` is used, but `range()` function is not suitable for string arguments. \n\nInstead, the `for` loop should iterate over the range of the length of the sentence (`range(len(sentence))`), as this will correctly provide the indices of each character in the sentence.\n\nHere is the corrected code snippet:\n\n```python\ndef consonant_locations(sentence):\n    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']\n    locations = []\n    for i in range(len(sentence)):\n        if sentence[i] in consonants:\n            locations.append(i)\n    return locations\nprint(consonant_locations('synchronization'))\n```\n\nThis will return all locations of consonants in the provided string correctly. However, it doesn't handle uppercase consonants, punctuation marks, numbers, and spaces. To make it handle these edge cases as well, we can revise the code as:\n\n```python\ndef consonant_locations(sentence):\n    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']\n    consonants += [c.upper() for c in consonants]  # handle uppercase\n    locations = []\n    for i in range(len(sentence)):\n        if sentence[i] in consonants:\n            locations.append(i)\n    return locations\nprint(consonant_locations('Synchronization.'))  # handle uppercase letters and punctuation\n```\n\nThis revised code correctly identifies the locations of all consonants in the provided string, keeps track of the original case, and ignores punctuation, spaces, and numbers."""

# yizhongw/self_instruct
#instruction = """Make a list of 10 ways to help students improve their study skills.\n\nOutput:"""
#output= """ 1. Make a schedule for studying and stick to it.\n2. Study in the same place every time.\n3. Set goals for yourself.\n4. Take breaks when you need them.\n5. Don't cram before an exam.\n6. Get enough sleep.\n7. Eat healthy food.\n8. Exercise regularly.\n9. Find a study partner.\n10. Reward yourself after completing a task."Make a list of 10 ways to help students improve their study skills.\n\nOutput:"""

# tatsu-lab/alpaca
#instruction = """Give three tips for staying healthy."""
#output = """1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.', 'text': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."""
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")

tokenized = tokenizer([instruction+'\n'+output], return_tensors="pt", max_length=1024, truncation=True)

input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
labels = input_ids.clone()
label_len = tokenizer([output], return_tensors="pt", max_length=1024, truncation=True, add_special_tokens=False)['input_ids'].shape[-1]
labels[0,:-label_len] = 1
print(tokenizer.decode(input_ids[0]))
print('###############')
print(tokenizer.decode(labels[0]))
labels[0,:-label_len] = -100
model.train()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(loss)