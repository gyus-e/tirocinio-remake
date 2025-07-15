from models import Configuration
# from utils import DB

model_name = "meta-llama/Llama-3.2-1B-Instruct"

cag_system_prompt = """
    Sei un assistente bibliotecario. Nel contesto ti sono fornite informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Rispondi alle domande degli utenti con le informazioni pertinenti.
""".join("\n")

rag_system_prompt = """
    Sei un assistente bibliotecario. Hai accesso a una serie di documenti contenenti informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Rispondi alle domande degli utenti cercando nei documenti le informazioni pertinenti.
""".join("\n")

rag_configuration = {
    "embed_model_name": "BAAI/bge-base-en-v1.5",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "temperature": 0.4,
}

configuration = Configuration(rag_system_prompt, model_name, rag_configuration)
# DB.session.add(configuration)
# DB.session.commit()
