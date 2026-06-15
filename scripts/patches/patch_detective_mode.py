import re

with open("engine/retrieval/detective_mode.py", "r") as f:
    content = f.read()

# _linguist_query_expansion
content = content.replace(
    'def _linguist_query_expansion(query: str, trace_id: str) -> Tuple[dict, dict]:',
    'def _linguist_query_expansion(query: str, trace_id: str, model_override: str | None = None) -> Tuple[dict, dict]:'
)
content = content.replace(
    'llm = rag_chain.get_llm()\n    try:\n        resp = llm.invoke(LINGUIST_EXPANSION_PROMPT.format(query=query.strip()))',
    'llm = rag_chain.get_llm(model_override)\n    try:\n        resp = llm.invoke(LINGUIST_EXPANSION_PROMPT.format(query=query.strip()))'
)

# _check_missing_info
content = content.replace(
    '    history: Optional[List[dict]],\n    trace_id: str,\n) -> Tuple[',
    '    history: Optional[List[dict]],\n    trace_id: str,\n    model_override: str | None = None,\n) -> Tuple['
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = llm.invoke(\n            MISSING_INFO_PROMPT.format(context=context_str, query=query.strip())',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = llm.invoke(\n            MISSING_INFO_PROMPT.format(context=context_str, query=query.strip())'
)

# _internal_knowledge_fallback
content = content.replace(
    'def _internal_knowledge_fallback(query: str, trace_id: str) -> Tuple[str, dict]:',
    'def _internal_knowledge_fallback(query: str, trace_id: str, model_override: str | None = None) -> Tuple[str, dict]:'
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = llm.invoke(\n            INTERNAL_KNOWLEDGE_FALLBACK_PROMPT.format(query=query.strip())',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = llm.invoke(\n            INTERNAL_KNOWLEDGE_FALLBACK_PROMPT.format(query=query.strip())'
)

# _synthesis_causality_skeptic_flip
content = content.replace(
    '    source_docs: List[Document],\n    trace_id: str,\n) -> Tuple[str, dict]:',
    '    source_docs: List[Document],\n    trace_id: str,\n    model_override: str | None = None,\n) -> Tuple[str, dict]:'
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = llm.invoke(\n            CAUSALITY_SYNTHESIS_PROMPT.format(',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = llm.invoke(\n            CAUSALITY_SYNTHESIS_PROMPT.format('
)

# _synthesis_partial_analysis
content = content.replace(
    '    data_pct: int,\n    trace_id: str,\n) -> Tuple[str, dict]:',
    '    data_pct: int,\n    trace_id: str,\n    model_override: str | None = None,\n) -> Tuple[str, dict]:'
)
content = content.replace(
    '    llm = rag_chain.get_llm()\n    try:\n        resp = llm.invoke(\n            PARTIAL_ANALYSIS_PROMPT.format(',
    '    llm = rag_chain.get_llm(model_override)\n    try:\n        resp = llm.invoke(\n            PARTIAL_ANALYSIS_PROMPT.format('
)

# invoke_detective_qa
content = content.replace(
    '    trace_id: Optional[str] = None,\n) -> dict:',
    '    trace_id: Optional[str] = None,\n    model_override: str | None = None,\n) -> dict:'
)

content = content.replace(
    '_linguist_query_expansion(query, trace_id)',
    '_linguist_query_expansion(query, trace_id, model_override)'
)

content = content.replace(
    '_check_missing_info(expanded_query, history, trace_id)',
    '_check_missing_info(expanded_query, history, trace_id, model_override)'
)

content = content.replace(
    'invoke_agentic_qa(\n        expanded_query, history=history, trace_id=trace_id\n    )',
    'invoke_agentic_qa(\n        expanded_query, history=history, trace_id=trace_id, model_override=model_override\n    )'
)

content = content.replace(
    '_internal_knowledge_fallback(query, trace_id)',
    '_internal_knowledge_fallback(query, trace_id, model_override)'
)

content = content.replace(
    '_synthesis_causality_skeptic_flip(\n            result, source_documents, trace_id\n        )',
    '_synthesis_causality_skeptic_flip(\n            result, source_documents, trace_id, model_override\n        )'
)

content = content.replace(
    '            data_pct,\n            trace_id,\n        )',
    '            data_pct,\n            trace_id,\n            model_override,\n        )'
)

with open("engine/retrieval/detective_mode.py", "w") as f:
    f.write(content)
