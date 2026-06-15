import re

with open("engine/retrieval/agentic_workflow.py", "r") as f:
    content = f.read()

# _crag_eval
content = content.replace(
    'def _crag_eval(query: str, docs: List[Document], trace_id: str) -> Tuple[bool, dict]:',
    'def _crag_eval(query: str, docs: List[Document], trace_id: str, model_override: str | None = None) -> Tuple[bool, dict]:'
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = await asyncio.to_thread(llm.invoke, prompt.format(query=query))',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = await asyncio.to_thread(llm.invoke, prompt.format(query=query))'
)

# _crag_rewrite
content = content.replace(
    'def _crag_rewrite(query: str, trace_id: str) -> Tuple[List[str], dict]:',
    'def _crag_rewrite(query: str, trace_id: str, model_override: str | None = None) -> Tuple[List[str], dict]:'
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = await asyncio.to_thread(llm.invoke, prompt.format(query=query))',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = await asyncio.to_thread(llm.invoke, prompt.format(query=query))'
)

# _cove_verify
content = content.replace(
    'def _cove_verify(query: str, context: str, response: str, trace_id: str) -> Tuple[bool, dict]:',
    'def _cove_verify(query: str, context: str, response: str, trace_id: str, model_override: str | None = None) -> Tuple[bool, dict]:'
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = await asyncio.to_thread(\n            llm.invoke, prompt.format(context=safe_context, response=safe_response)\n        )',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = await asyncio.to_thread(\n            llm.invoke, prompt.format(context=safe_context, response=safe_response)\n        )'
)

# invoke_agentic_qa
content = content.replace(
    '    trace_id: Optional[str] = None,\n) -> dict:',
    '    trace_id: Optional[str] = None,\n    model_override: str | None = None,\n) -> dict:'
)

content = content.replace(
    '_crag_eval(query, initial_docs, trace_id)',
    '_crag_eval(query, initial_docs, trace_id, model_override)'
)

content = content.replace(
    '_crag_rewrite(query, trace_id)',
    '_crag_rewrite(query, trace_id, model_override)'
)

content = content.replace(
    '_cove_verify(query, context_str, result, trace_id)',
    '_cove_verify(query, context_str, result, trace_id, model_override)'
)

content = content.replace(
    'base_res = await asyncio.to_thread(\n        rag_chain.invoke_qa, query, history=history\n    )',
    'base_res = await asyncio.to_thread(\n        rag_chain.invoke_qa, query, history=history, model_override=model_override\n    )'
)

with open("engine/retrieval/agentic_workflow.py", "w") as f:
    f.write(content)
