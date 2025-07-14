from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from environ import VECTOR_STORE_DIR

class Index:
    def __init__(self, documents: list[Document]):
        self._index: BaseIndex = VectorStoreIndex.from_documents(documents)

        print("Index created. Persisting to storage...")
        self._index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
        print("Index persisted to storage.")

    def index(self) -> BaseIndex:
        return self._index

    def load_index(self):
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
        self._index = load_index_from_storage(storage_context)
        print("Index loaded from storage.")
