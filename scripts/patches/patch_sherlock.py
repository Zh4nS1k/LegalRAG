import re

with open("ai_service/api/api.py", "r") as f:
    content = f.read()

content = content.replace(
    'sherlock = sherlock_engine.SherlockEngine()',
    'sherlock = sherlock_engine.SherlockEngine(model_override=body.model)'
)

with open("ai_service/api/api.py", "w") as f:
    f.write(content)

with open("ai_service/retrieval/sherlock_engine.py", "r") as f:
    content = f.read()

content = content.replace(
    'def __init__(self, max_iterations: int = 3):',
    'def __init__(self, max_iterations: int = 3, model_override: str | None = None):'
)

content = content.replace(
    'self.max_iterations = max_iterations',
    'self.max_iterations = max_iterations\n        self.model_override = model_override'
)

content = content.replace(
    'rag_chain.get_llm',
    'lambda: rag_chain.get_llm(self.model_override)'
)

with open("ai_service/retrieval/sherlock_engine.py", "w") as f:
    f.write(content)
