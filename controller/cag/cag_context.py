def build_cag_context(system_prompt: str, document_texts: list[str]) -> str:
    return f"""
    <|system|>
    {system_prompt}
    <|user|>
    Contesto:
    {document_texts}
    Domanda:
    """
