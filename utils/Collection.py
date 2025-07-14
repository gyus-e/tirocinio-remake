from llama_index.core import Document, SimpleDirectoryReader
from environ import DOCUMENTS_DIR


class Collection:
    def __init__(self):
        print("Loading documents from directory...")
        self._documents = SimpleDirectoryReader(input_dir=DOCUMENTS_DIR).load_data()
        print(f"Loaded {len(self._documents)} documents.")

    def documents(self) -> list[Document]:
        return self._documents
