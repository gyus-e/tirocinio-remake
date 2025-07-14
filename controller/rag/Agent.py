from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import Settings
from .AgentTools import AgentTools
from models import Configuration

class Agent:
    def __init__(self, configuration: Configuration, query_engine: BaseQueryEngine, with_context: bool = True):
        self._agent_tools = AgentTools(query_engine)

        self._agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[self._agent_tools.search_documents],
            llm=Settings.llm,
            system_prompt=configuration.system_prompt,
        )

        self._context = Context(self._agent) if with_context else None
    
    def reset_context(self):
        self._context = Context(self._agent)

    def agent(self):
        return self._agent
    
    def context(self):
        return self._context