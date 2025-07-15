from llama_index.core.base.base_query_engine import BaseQueryEngine


class AgentTools:
    def __init__(self, query_engine: BaseQueryEngine):
        self.query_engine = query_engine

    async def search_documents(self, query: str) -> str:
        """Useful for answering natural language questions about the content of the documents."""
        print(f"Using search_documents with query: {query}")
        response = await self.query_engine.aquery(query)
        return str(response)
