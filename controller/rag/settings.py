import os
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models import Configuration
from utils import LLM
from environ import HF_HOME, EMBED_MODEL_DIR


def initialize_settings(config: Configuration, llm: LLM | None = None):
    rag_settings = config.rag_configuration
    if not rag_settings:
        raise ValueError("RAG configuration is missing in the provided configuration.")

    if not llm:
        model_name = rag_settings.get("model_name")
        if not model_name:
            raise ValueError("Model name is required in the RAG configuration.")
        llm = LLM(model_name)

    embed_model_name: str = rag_settings.get(
        "embed_model_name", "BAAI/bge-base-en-v1.5"
    )
    chunk_size: int = rag_settings.get("chunk_size", 512)
    chunk_overlap: int = rag_settings.get("chunk_overlap", 50)

    temperature: float = rag_settings.get("temperature", 0.0)
    top_k: int | None = rag_settings.get("top_k", None)
    top_p: float | None = rag_settings.get("top_p", None)

    kwargs = {"temperature": temperature} if temperature > 0 else {"do_sample": True}
    kwargs = {"top_k": top_k} if top_k else kwargs
    kwargs = {"top_p": top_p} if top_p else kwargs

    Settings.llm = HuggingFaceLLM(
        model=llm.model(),
        tokenizer=llm.tokenizer(),
        # context_window=CONTEXT_WINDOW if CONTEXT_WINDOW else DEFAULT_CONTEXT_WINDOW,
        generate_kwargs=kwargs,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        cache_folder=os.path.join(HF_HOME, EMBED_MODEL_DIR) if HF_HOME else None,
    )

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    # For testing, comment all the above and uncomment the following lines. Set up your OpenAI API key in the .env file.
    # Settings.llm = OpenAI(
    #     model="gpt-3.5-turbo",
    #     api_key=OPENAI_API_KEY,
    # )
