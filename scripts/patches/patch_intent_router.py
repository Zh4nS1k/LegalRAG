import re

with open("ai_service/api/api.py", "r") as f:
    content = f.read()

content = content.replace(
    'intent_router.route_query, body.query, history=body.history',
    'intent_router.route_query, body.query, history=body.history, model_override=body.model'
)

with open("ai_service/api/api.py", "w") as f:
    f.write(content)

with open("ai_service/retrieval/intent_router.py", "r") as f:
    content = f.read()

content = content.replace(
    'def route_query(query: str, history: Optional[List[dict]] = None) -> dict:',
    'def route_query(query: str, history: Optional[List[dict]] = None, model_override: str | None = None) -> dict:'
)

content = content.replace(
    'def _llm_router(query: str, history: str) -> Tuple[str, dict]:',
    'def _llm_router(query: str, history: str, model_override: str | None = None) -> Tuple[str, dict]:'
)

content = content.replace(
    'llm = get_llm()',
    'llm = get_llm(model_override)'
)

content = content.replace(
    '_llm_router(query, history_str)',
    '_llm_router(query, history_str, model_override)'
)

with open("ai_service/retrieval/intent_router.py", "w") as f:
    f.write(content)
