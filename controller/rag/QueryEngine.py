from llama_index.core.indices.base import BaseIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine

class QueryEngine:
    def __init__(self, index: BaseIndex):
        print("Creating query engine...")
        self._query_engine = index.as_query_engine()
        print("Query engine created.")

    def query_engine(self) -> BaseQueryEngine:
        return self._query_engine