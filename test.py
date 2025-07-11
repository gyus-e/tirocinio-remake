import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from environ import STORAGE, CACHE_NAME
from models import Configuration
from controller import cag


model_name = "meta-llama/Llama-3.2-1B-Instruct"

system_prompt = """
You are an assistant who provides concise factual answers.
"""

configuration = Configuration(system_prompt, model_name)

prompt = f"""
<|system|>
{configuration.system_prompt}
<|user|>
Context:
My best friend, Claudio, lives near Rome, in the city of Ardea.
He was born in January 2000.
He studied at Liceo Scientifico and currently works at a pharmaceutical company.
He has two cats: Venere and Flora.
He is married to a British girl named Arturia.
He is interested in animation and video-editing.
Arturia is an 18 year old British girl.
She is a glutton and loves Italian cuisine, but also junk food such as hamburgers.
She is blonde and has green eyes.
She is really into chivalry and medieval geopolitics.
Sometimes she goes to Burger King just to wear the cardboard crown they give and pretend she is the King of England.
Question:
"""



model = AutoModelForCausalLM.from_pretrained(configuration.model_name)
tokenizer = AutoTokenizer.from_pretrained(configuration.model_name)
device = Accelerator().device


cache_path = os.path.join(STORAGE, CACHE_NAME)
if not os.path.exists(cache_path):
    cache = cag.create_kv_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt.strip(),
    )
    cag.save_cache(cache, storage=STORAGE, cache_name=CACHE_NAME)



questions = [
    "Where does Claudio live?",
    "Where is my best friend's wife from?",
    "What are the names of my best friend's cats?",
    "What does Arturia look like?",
    "Do they have any children?",
    ]
for question in questions:
    print(f"Question: {question}")
    cache = torch.load(cache_path, weights_only=False)
    test_answer = cag.get_answer(question, tokenizer, model, device, cache)
    cag.clean_up_cache(cache)
    print(f"Answer: {test_answer}\n")
