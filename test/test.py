import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from environ import STORAGE, CACHE_NAME, VECTOR_STORE_DIR
from controller import cag, rag
from utils import Collection
from .configuration_mock import configuration
from .questions_mock import questions

torch.set_grad_enabled(False) 
documents = None

# CAG initialization
device = Accelerator().device
model = AutoModelForCausalLM.from_pretrained(configuration.model_name, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(configuration.model_name)


cache_path = os.path.join(STORAGE, CACHE_NAME)
if not os.path.exists(cache_path):
    documents = Collection().documents()

    document_texts = [doc.text for doc in documents]
    cag_prompt = cag.build_cag_context(configuration.system_prompt, document_texts)

    cache = cag.create_kv_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=cag_prompt,
    )
    cag.save_cache(cache, storage=STORAGE, cache_name=CACHE_NAME)


# RAG initialization
rag.initialize_settings(configuration)

if not os.path.exists(VECTOR_STORE_DIR):
    documents = Collection().documents() if not documents else documents
    rag_index = rag.Index.from_documents(documents)
    rag_index.persist(VECTOR_STORE_DIR)
    index = rag_index.index()
else:
    index = rag.Index.from_storage(VECTOR_STORE_DIR).index()

query_engine = rag.QueryEngine(index=index).query_engine()
rag_agent = rag.Agent(query_engine=query_engine, system_prompt=configuration.system_prompt)
agent = rag_agent.agent()
context = rag_agent.context()

# Test
async def test():
    print("\n\tCAG\n")
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")

        cache = torch.load(cache_path, weights_only=False)
        cag_answer = cag.get_answer(question, tokenizer, model, device, cache)
        cag.clean_up_cache(cache)
        print(f"CAG: {cag_answer}\n")

    print("\t*********************************")
    
    print("\n\tRAG\n")    
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")

        rag_answer = await agent.run(question, ctx=context)
        print(f"RAG: {rag_answer}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test())