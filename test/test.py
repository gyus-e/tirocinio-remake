import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from environ import STORAGE, CACHE_NAME, VECTOR_STORE_DIR
from controller import cag, rag
from utils import Collection
from .configuration_mock import configuration
from .questions_mock import questions

documents = None

# CAG initialization
device = Accelerator().device
model = AutoModelForCausalLM.from_pretrained(configuration.model_name, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(configuration.model_name)

cache_path = os.path.join(STORAGE, CACHE_NAME)
if not os.path.exists(cache_path):
    documents = Collection().documents()

    document_texts = [doc.text for doc in documents]
    cag_prompt = cag.build_cag_context("Sei l'assistente bibliotecario della Biblioteca Pontaniana di Napoli. Rispondi brevemente alle domande basandoti sul contesto fornito.", document_texts)

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
agent = rag.Agent(query_engine=query_engine, system_prompt=configuration.system_prompt).agent()


# Test
async def test():
    for question in questions:
        print("\n\t****************************************\n")
        print(f"Question: {question}")
        
        cache = torch.load(cache_path, weights_only=False)
        cag_answer = cag.get_answer(question, tokenizer, model, device, cache)
        cag.clean_up_cache(cache)
        print(f"CAG Answer: {cag_answer}\n")
        
        print("\n")
        
        rag_answer = await agent.run(question)
        print(f"RAG Answer: {rag_answer}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test())