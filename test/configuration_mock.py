from models import Configuration
# from utils import DB

model_name = "meta-llama/Llama-3.2-3B-Instruct"

system_prompt = """
Sei l'assistente bibliotecario della Biblioteca Pontaniana di Napoli.
Hai accesso a dei documenti che contengono informazioni sulle opere conservate nella biblioteca.
Rispondi alle domande basandoti esclusivamente sul contenuto dei documenti forniti.
Se non hai abbastanza informazioni, rispondi "Non lo so".
Non fornire mai informazioni che non sono presenti nei documenti.
"""

rag_configuration = {
    "embed_model_name": "BAAI/bge-base-en-v1.5",
    "chunk_size": 1024,
    "chunk_overlap": 128,
    "temperature": 0.4,
}

configuration = Configuration(system_prompt, model_name, rag_configuration)
# DB.session.add(configuration)
# DB.session.commit()