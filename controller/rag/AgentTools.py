from llama_index.core.base.base_query_engine import BaseQueryEngine

class AgentTools:
    def __init__(self, query_engine: BaseQueryEngine):
        self.query_engine = query_engine

    async def search_documents(self, query: str) -> str:
        """Cerca nei documenti il contesto utile a rispondere alla domanda."""
        response = await self.query_engine.aquery(query)
        return str(response)