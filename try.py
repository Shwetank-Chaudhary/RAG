from template import prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login
import os 
from dotenv import load_dotenv

login(os.getenv("hugging_face_login"))

model_id = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype = torch.float16,
                                             device_map = "auto")

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

template = prompt("bbbibbkll","how are you")

output = text_generator(prompt, do_sample=False)[0]["generated_text"]

# Optional: Strip out prompt from result if needed
print("Output:", output.replace(prompt, "").strip())

# formated_promt = f"<s>[INST]{template}[/INST]"

# inputs = tokenizer(formated_promt, return_tensors = "pt").to(model.device)

# outputs = model.generate(**inputs,max_new_tokens=512)

# print(tokenizer.decode(outputs[0], skip_special_tokens = True))