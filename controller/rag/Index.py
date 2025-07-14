from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from environ import VECTOR_STORE_DIR

class Index:
    def __init__(self, index: BaseIndex):
        self._index = index
    
    @classmethod
    def from_documents(cls, documents: list[Document]):
        index = VectorStoreIndex.from_documents(documents)
        return cls(index)
    
    @classmethod
    def from_storage(cls, persist_dir: str = VECTOR_STORE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("Index loaded from storage.")
        return cls(index)

    def index(self) -> BaseIndex:
        return self._index

    def persist(self, persist_dir: str = VECTOR_STORE_DIR):
        self._index.storage_context.persist(persist_dir=persist_dir)
        print("Index persisted to storage.")
        return self
