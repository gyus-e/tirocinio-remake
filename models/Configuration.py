from utils import DB


__all__ = ["Configuration"]


class Configuration(DB.Model):
    config_id = DB.Column(DB.Integer, primary_key=True, autoincrement=True)
    system_prompt = DB.Column(DB.String, nullable=False)
    model_name = DB.Column(DB.String, nullable=False)
    rag_configuration = DB.Column(DB.JSON, nullable=True)

    def __init__(self, system_prompt: str, model_name: str, rag_configuration: dict | None = None):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.rag_configuration = rag_configuration