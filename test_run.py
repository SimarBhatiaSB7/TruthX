import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

llama2chat_with_truthx = "ICTNLP/Llama-2-7b-chat-TruthX"
tokenizer = AutoTokenizer.from_pretrained(llama2chat_with_truthx, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(llama2chat_with_truthx, trust_remote_code=True,torch_dtype=torch.float16).cuda()

question = "What are the benefits of eating an apple a day?"
encoded_inputs = tokenizer(question, return_tensors="pt")["input_ids"]
outputs = model.generate(encoded_inputs.cuda())[0, encoded_inputs.shape[-1] :]
outputs_text = tokenizer.decode(outputs, skip_special_tokens=True).strip()
print(outputs_text)