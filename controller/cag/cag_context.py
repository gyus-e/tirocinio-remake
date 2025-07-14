def build_cag_context(system_prompt: str, document_texts: list[str]) -> str:
    return f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|start_header_id|>user<|end_header_id|>
    Context:
    {document_texts}
    Question:
    """
