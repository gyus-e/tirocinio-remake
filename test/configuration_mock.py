from models import Configuration
# from utils import DB

model_name = "meta-llama/Llama-3.2-1B-Instruct"

system_prompt = """
You are a helpful assistent that provides concise and accurate answers about the provided context.
"""

rag_configuration = {
    "embed_model_name": "BAAI/bge-base-en-v1.5",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "temperature": 0.4,
}

configuration = Configuration(system_prompt, model_name, rag_configuration)
# DB.session.add(configuration)
# DB.session.commit()